"""
In-process implementations of `ChunkStore`, `UploadStore`, `JobQueue` and
`EmbeddingCache`.

Stand-ins for pgvector, Blob Storage, Service Bus and the Postgres embedding
cache, so the indexing pipeline can be exercised without any of them. They
honour the same contracts as the cloud adapters — `tests/port_contracts/` runs
one suite against both — including the parts that only matter when something
goes wrong: settlement, redelivery and dead-lettering.
"""
from __future__ import annotations

import asyncio
import math

from contracts.indexing import ChunkUpload, IndexJobMessage, SearchHit
from core.index.ports import IndexedChunk, StoredFile, UploadNotFoundError, chunk_id


def _cosine_similarity(one: list[float], other: list[float]) -> float:
    """1.0 for identical directions, 0.0 for orthogonal — what pgvector's distance inverts to."""
    dot = sum(a * b for a, b in zip(one, other, strict=True))
    size = math.sqrt(sum(a * a for a in one)) * math.sqrt(sum(b * b for b in other))
    return dot / size if size else 0.0


class InMemoryChunkStore:
    """
    Stands in for Postgres with pgvector: chunks in a dict, searched by cosine
    similarity.

    Atomic for free — a dict mutation cannot be interrupted — so it satisfies
    the port's "a search sees the index from before the job or after it" without
    the transaction the real store needs.
    """

    def __init__(self) -> None:
        #: project id → chunk id → the chunk.
        self._projects: dict[str, dict[str, IndexedChunk]] = {}

    def __len__(self) -> int:
        return sum(len(chunks) for chunks in self._projects.values())

    async def manifest(self, project_id: str) -> dict[str, StoredFile]:
        by_file: dict[str, list[IndexedChunk]] = {}
        for chunk in self._projects.get(project_id, {}).values():
            by_file.setdefault(chunk.file_path, []).append(chunk)

        manifest = {}
        for path, chunks in by_file.items():
            chunks.sort(key=lambda chunk: chunk.start_line)
            manifest[path] = StoredFile(
                # A file's chunks are written by one job and share its hash.
                file_hash=chunks[0].file_hash,
                ids=tuple(chunk_id(chunk.file_path, chunk.start_line) for chunk in chunks),
            )
        return manifest

    async def apply(
        self,
        project_id: str,
        *,
        upserts: list[IndexedChunk],
        delete: list[str],
        reset: bool = False,
    ) -> int:
        chunks = self._projects.setdefault(project_id, {})
        if reset:
            chunks.clear()

        # Deletes run before upserts, as they do in the Postgres transaction, so
        # an id in both lists ends up written rather than dropped.
        removed = sum(1 for key in delete if chunks.pop(key, None) is not None)
        for chunk in upserts:
            chunks[chunk.id] = chunk
        return removed

    async def search(self, project_id: str, vector: list[float], n: int) -> list[SearchHit]:
        scored = sorted(
            (
                (_cosine_similarity(vector, chunk.vector), chunk)
                for chunk in self._projects.get(project_id, {}).values()
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            SearchHit(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                file_hash=chunk.file_hash,
                score=score,
            )
            for score, chunk in scored[:n]
        ]


class InMemoryUploadStore:
    """Stands in for Blob Storage: uploads by reference, in a dict."""

    def __init__(self) -> None:
        self._uploads: dict[str, ChunkUpload] = {}

    def __len__(self) -> int:
        return len(self._uploads)

    async def put(self, upload_ref: str, upload: ChunkUpload) -> None:
        self._uploads[upload_ref] = upload

    async def get(self, upload_ref: str) -> ChunkUpload:
        try:
            return self._uploads[upload_ref]
        except KeyError:
            raise UploadNotFoundError(upload_ref) from None

    async def delete(self, upload_ref: str) -> None:
        self._uploads.pop(upload_ref, None)


class InMemoryEmbeddingCache:
    """Vectors by `(model, digest)` in a dict. Unbounded: for tests and short-lived use."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str], list[float]] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    async def get(self, model: str, digests: list[str]) -> dict[str, list[float]]:
        return {
            digest: self._vectors[(model, digest)]
            for digest in digests
            if (model, digest) in self._vectors
        }

    async def put(self, model: str, vectors: dict[str, list[float]]) -> None:
        for digest, vector in vectors.items():
            self._vectors[(model, digest)] = list(vector)


class _InMemoryDelivery:
    def __init__(
        self, queue: InMemoryJobQueue, message: IndexJobMessage, delivery_count: int
    ) -> None:
        self.message = message
        self.delivery_count = delivery_count
        self._queue = queue
        self._settled = False

    async def complete(self) -> None:
        self._settle()

    async def abandon(self) -> None:
        self._settle()
        await self._queue._redeliver(self.message, self.delivery_count)

    async def dead_letter(self, reason: str) -> None:
        self._settle()
        self._queue.dead_letters.append((self.message, reason))

    def _settle(self) -> None:
        # Service Bus refuses to settle a message twice. So does this, so a
        # double settlement is a bug found by a unit test, not in Azure.
        if self._settled:
            raise RuntimeError(f"The delivery of job {self.message.job_id} is already settled.")
        self._settled = True


class InMemoryJobQueue:
    """
    Stands in for Service Bus: in-process and FIFO, with its failure semantics.

    `max_delivery_count` mirrors the queue setting of the same name: a message
    abandoned on its last allowed delivery is dead-lettered instead of coming
    back, and kept in `dead_letters` with the reason, for inspection.
    """

    def __init__(self, *, max_delivery_count: int = 5) -> None:
        self._queue: asyncio.Queue[tuple[IndexJobMessage, int]] = asyncio.Queue()
        self._max_delivery_count = max_delivery_count
        self.dead_letters: list[tuple[IndexJobMessage, str]] = []

    async def enqueue(self, message: IndexJobMessage) -> None:
        await self._queue.put((message, 1))

    async def receive(self) -> _InMemoryDelivery:
        message, delivery_count = await self._queue.get()
        return _InMemoryDelivery(self, message, delivery_count)

    async def _redeliver(self, message: IndexJobMessage, delivery_count: int) -> None:
        if delivery_count >= self._max_delivery_count:
            self.dead_letters.append((message, "MaxDeliveryCountExceeded"))
        else:
            await self._queue.put((message, delivery_count + 1))
