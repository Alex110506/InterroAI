"""
SemanticIndex — the runtime's only view of the code index.

Four operations, each shaped like the HTTP call it becomes in the cloud build:

  sync(SyncRequest)      → SyncResult      POST /projects/{id}/sync
  upload(ChunkUpload)    → job id          PUT to Blob Storage (SAS) + POST /projects/{id}/jobs
  job_events(job_id)     → IndexEvent …    GET  /jobs/{id}/events  (server-sent events)
  search(SearchRequest)  → [SearchHit]     POST /projects/{id}/search

A search returns *where* the code is — path, line range, and the hash of the
file it was indexed from — never the code itself. The index stores no source
text; the caller reads the lines from its own disk and uses the hash to tell
whether they may have moved since.

`LocalSemanticIndex` implements all four in-process, assembled from the same
ports the cloud side uses (`core/index/ports.py`) with local adapters plugged
in: Chroma on disk as the `ChunkStore`, the optional Redis as the
`EmbeddingCache`, in-memory stand-ins for Blob Storage and Service Bus, and a
background task as the worker. The cloud worker and API assemble the same
pieces — `diff_manifest`, `run_job` — around cloud adapters instead.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import uuid4

from contracts.indexing import (
    ChunkUpload,
    IndexEvent,
    IndexJobMessage,
    SearchHit,
    SearchRequest,
    SyncRequest,
    SyncResult,
)
from core.index.adapters.chroma import ChromaChunkStore
from core.index.adapters.memory import InMemoryJobQueue, InMemoryUploadStore
from core.index.adapters.redis_cache import RedisEmbeddingCache
from core.index.embeddings import embed_texts
from core.index.indexer import run_job
from core.index.manifest_diff import diff_manifest
from core.index.ports import ChunkStore, EmbeddingCache, JobQueue, UploadStore

logger = logging.getLogger(__name__)


class SemanticIndex(Protocol):
    async def sync(self, request: SyncRequest) -> SyncResult:
        """Diff the client's files against what the index holds."""
        ...

    async def upload(self, upload: ChunkUpload) -> str:
        """Hand an upload to the worker. Returns as soon as it is queued."""
        ...

    def job_events(self, job_id: str) -> AsyncIterator[IndexEvent]:
        """A job's progress, ending with its `done` or `error` event."""
        ...

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """The closest chunks to a natural-language query, best first."""
        ...


class LocalSemanticIndex:
    """
    The whole indexing service, inside this process.

    One handle per caller: its queue, its event buffers and its worker tasks
    belong to the event loop that first uses them, so sharing one handle across
    loops would fail in ways that have nothing to do with indexing.

    Every part is injectable — which is how tests hand in a spy — and defaults
    to its local adapter.
    """

    def __init__(
        self,
        *,
        store: ChunkStore | None = None,
        cache: EmbeddingCache | None = None,
        uploads: UploadStore | None = None,
        queue: JobQueue | None = None,
    ) -> None:
        # `is None`, not `or`: an empty in-memory store is falsy.
        self._store = ChromaChunkStore() if store is None else store
        self._cache = RedisEmbeddingCache() if cache is None else cache
        self._uploads = InMemoryUploadStore() if uploads is None else uploads
        self._queue = InMemoryJobQueue() if queue is None else queue
        #: Per-job progress, buffered until the client reads it. `None` marks
        #: the end of a job's stream.
        self._events: dict[str, asyncio.Queue[IndexEvent | None]] = {}
        # asyncio keeps only weak references to tasks; without this set a
        # worker could be garbage-collected mid-job.
        self._workers: set[asyncio.Task] = set()

    # ── Sync ─────────────────────────────────────────────────────────────────

    async def sync(self, request: SyncRequest) -> SyncResult:
        indexed = {} if request.force else await self._store.manifest(request.project_id)
        return diff_manifest(
            request.files,
            {path: stored.file_hash for path, stored in indexed.items()},
            force=request.force,
        )

    # ── Upload and job progress ──────────────────────────────────────────────

    async def upload(self, upload: ChunkUpload) -> str:
        job_id = uuid4().hex
        upload_ref = f"uploads/{job_id}"

        await self._uploads.put(upload_ref, upload)
        self._events[job_id] = asyncio.Queue()
        await self._queue.enqueue(
            IndexJobMessage(job_id=job_id, project_id=upload.project_id, upload_ref=upload_ref)
        )

        worker = asyncio.create_task(self._consume_one())
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        return job_id

    async def job_events(self, job_id: str) -> AsyncIterator[IndexEvent]:
        events = self._events.get(job_id)
        if events is None:
            yield IndexEvent(step="error", message=f"Unknown index job: {job_id}")
            return
        try:
            while (event := await events.get()) is not None:
                yield event
        finally:
            self._events.pop(job_id, None)

    async def _consume_one(self) -> None:
        """The worker: take one delivery, fetch its upload, run the job, settle."""
        delivery = await self._queue.receive()
        job_id = delivery.message.job_id
        upload_ref = delivery.message.upload_ref
        try:
            upload = await self._uploads.get(upload_ref)
            async for event in run_job(upload, store=self._store, cache=self._cache):
                self._publish(job_id, event)
            # Claim-check: the upload is deleted once its job has run, as the
            # cloud worker deletes the blob.
            await self._uploads.delete(upload_ref)
            await delivery.complete()
        except Exception as exc:  # noqa: BLE001
            # `run_job` already turns failures into events, so this is a defect
            # in the worker itself. Redelivery will not fix it, so the message
            # is dead-lettered — and still reported, or the client would wait
            # on a stream that never ends.
            logger.exception("Index worker failed outside the job")
            self._publish(job_id, IndexEvent(step="error", message=str(exc)))
            await delivery.dead_letter(str(exc))
        finally:
            self._publish(job_id, None)

    def _publish(self, job_id: str, event: IndexEvent | None) -> None:
        events = self._events.get(job_id)
        # A client that stopped listening has already dropped its buffer; the
        # job still runs to completion, it just has no audience.
        if events is not None:
            events.put_nowait(event)

    # ── Search ───────────────────────────────────────────────────────────────

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        [vector] = await embed_texts([request.query], cache=self._cache)
        return await self._store.search(request.project_id, vector, request.n)
