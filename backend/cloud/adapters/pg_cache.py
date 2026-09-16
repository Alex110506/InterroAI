"""
The cloud `EmbeddingCache`: a Postgres table beside the index it feeds.

Shared by every tenant on purpose. It is keyed by a hash of the chunk's text,
so identical code in two projects is embedded — and paid for — once; and it
holds only vectors and hashes, with no text and no link to a project or user.
Replacing Redis with it is also a cost decision: it lives in the database the
worker already has.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from cloud.db.models import EmbeddingCacheEntry
from cloud.db.session import ScopeFactory, as_float_list


class PostgresEmbeddingCache:
    def __init__(self, scope: ScopeFactory) -> None:
        self._scope = scope

    async def get(self, model: str, digests: list[str]) -> dict[str, list[float]]:
        if not digests:
            return {}
        statement = select(EmbeddingCacheEntry.content_hash, EmbeddingCacheEntry.embedding).where(
            EmbeddingCacheEntry.model == model,
            EmbeddingCacheEntry.content_hash.in_(digests),
        )
        async with self._scope() as session:
            rows = (await session.execute(statement)).all()
        return {digest: as_float_list(vector) for digest, vector in rows}

    async def put(self, model: str, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        statement = (
            insert(EmbeddingCacheEntry)
            .values(
                [
                    {"model": model, "content_hash": digest, "embedding": vector}
                    for digest, vector in vectors.items()
                ]
            )
            # Identical text has an identical vector: a concurrent writer that
            # got there first wrote exactly this row.
            .on_conflict_do_nothing()
        )
        async with self._scope() as session:
            await session.execute(statement)
