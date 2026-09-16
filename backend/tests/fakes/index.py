"""
The indexing service, in this process.

The same `run_job` the Embed Worker runs in Azure, over the in-memory adapters
instead of pgvector, Blob Storage and Service Bus — so a test can drive the
whole of Phase 2, both halves, without any of them. The job runs during
`upload` and its events are replayed by `job_events`, which is all the runtime
half ever asks of a `SemanticIndex`.

Embedding is the one thing left to the caller: pass *embed* to answer queries
with a fixed vector, and patch `core.index.indexer.embed_batches` for the
worker's side.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from contracts.indexing import ChunkUpload, IndexEvent, SearchHit, SearchRequest, SyncRequest
from core.index.adapters.memory import InMemoryChunkStore, InMemoryEmbeddingCache
from core.index.embeddings import embed_texts
from core.index.indexer import run_job
from core.index.manifest_diff import diff_manifest

#: The vector every fake embedder here hands back. Cosine similarity against
#: itself is 1.0, so a search returns everything stored, best first.
VECTOR = [0.1, 0.2, 0.3, 0.4]


class InProcessIndex:
    """`SemanticIndex`, backed by dicts. Holds no source text, exactly as the real ones do not."""

    def __init__(self, *, embed=None) -> None:
        self.store = InMemoryChunkStore()
        self.cache = InMemoryEmbeddingCache()
        #: Every upload this index was handed, in order.
        self.uploads: list[ChunkUpload] = []
        self._embed = embed or embed_texts
        self._events: dict[str, list[IndexEvent]] = {}

    async def sync(self, request: SyncRequest):
        indexed = {} if request.force else await self.store.manifest(request.project_id)
        return diff_manifest(
            request.files,
            {path: stored.file_hash for path, stored in indexed.items()},
            force=request.force,
        )

    async def upload(self, upload: ChunkUpload) -> str:
        self.uploads.append(upload)
        job_id = uuid4().hex
        self._events[job_id] = [
            event async for event in run_job(upload, store=self.store, cache=self.cache)
        ]
        return job_id

    async def job_events(self, job_id: str) -> AsyncIterator[IndexEvent]:
        for event in self._events.pop(job_id, []):
            yield event

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        # The query is embedded on the index's side, as it is in the cloud: the
        # runtime sends a string and never embeds anything itself.
        vector = (await self._embed([request.query]))[0]
        return await self.store.search(request.project_id, vector, request.n)
