"""
The cloud `ChunkStore`: Postgres with pgvector.

Every call runs in a scope it opens itself (`cloud/db/session.py`), so one class
serves the worker, in a service scope, and the API, where a user scope lets
row-level security guarantee a search can only ever see that user's projects.

`apply` is one transaction: the reset, the prune and the upsert commit together
or not at all. That is what the Chroma adapter can only approximate by ordering
its writes, and what makes "a search sees the index from before the job or
after it" literally true.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import TypeVar

from sqlalchemy import delete as sql_delete
from sqlalchemy import select, text, tuple_
from sqlalchemy.dialects.postgresql import insert

from cloud.db.models import Chunk
from cloud.db.session import ScopeFactory
from contracts.indexing import SearchHit
from core.index.ports import IndexedChunk, StoredFile, chunk_id, split_chunk_id

#: Rows per statement: well inside Postgres's 65,535 bind-parameter limit.
_BATCH = 500

_T = TypeVar("_T")


def _batched(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class PostgresChunkStore:
    def __init__(self, scope: ScopeFactory) -> None:
        self._scope = scope

    async def manifest(self, project_id: str) -> dict[str, StoredFile]:
        statement = (
            select(Chunk.file_path, Chunk.start_line, Chunk.file_hash)
            .where(Chunk.project_id == uuid.UUID(project_id))
            .order_by(Chunk.file_path, Chunk.start_line)
        )
        async with self._scope() as session:
            rows = (await session.execute(statement)).all()

        hashes: dict[str, str] = {}
        ids: dict[str, list[str]] = {}
        for file_path, start_line, file_hash in rows:
            ids.setdefault(file_path, []).append(chunk_id(file_path, start_line))
            # A file's chunks are written by one job and share its hash.
            hashes.setdefault(file_path, file_hash)
        return {path: StoredFile(hashes[path], tuple(keys)) for path, keys in ids.items()}

    async def apply(
        self,
        project_id: str,
        *,
        upserts: list[IndexedChunk],
        delete: list[str],
        reset: bool = False,
    ) -> int:
        project = uuid.UUID(project_id)
        deleted = 0

        async with self._scope() as session:
            if reset:
                await session.execute(sql_delete(Chunk).where(Chunk.project_id == project))

            for keys in _batched([split_chunk_id(key) for key in delete], _BATCH):
                result = await session.execute(
                    sql_delete(Chunk).where(
                        Chunk.project_id == project,
                        tuple_(Chunk.file_path, Chunk.start_line).in_(keys),
                    )
                )
                deleted += result.rowcount

            for chunks in _batched(upserts, _BATCH):
                statement = insert(Chunk).values(
                    [
                        {
                            "project_id": project,
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "file_hash": chunk.file_hash,
                            "embedding": chunk.vector,
                        }
                        for chunk in chunks
                    ]
                )
                await session.execute(
                    statement.on_conflict_do_update(
                        index_elements=[Chunk.project_id, Chunk.file_path, Chunk.start_line],
                        set_={
                            "end_line": statement.excluded.end_line,
                            "file_hash": statement.excluded.file_hash,
                            "embedding": statement.excluded.embedding,
                        },
                    )
                )

        return deleted

    async def search(self, project_id: str, vector: list[float], n: int) -> list[SearchHit]:
        distance = Chunk.embedding.cosine_distance(vector)
        statement = (
            select(
                Chunk.file_path,
                Chunk.start_line,
                Chunk.end_line,
                Chunk.file_hash,
                distance.label("distance"),
            )
            .where(Chunk.project_id == uuid.UUID(project_id))
            .order_by(distance)
            .limit(n)
        )
        async with self._scope() as session:
            # The HNSW scan is approximate and the project filter is applied
            # after it, so a plain scan can come back with fewer than n rows
            # even when the project has more. Iterative scanning keeps going.
            await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
            rows = (await session.execute(statement)).all()

        return [
            SearchHit(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                file_hash=file_hash,
                score=1.0 - float(row_distance),
            )
            for file_path, start_line, end_line, file_hash, row_distance in rows
        ]
