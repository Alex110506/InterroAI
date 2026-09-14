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

`LocalSemanticIndex` implements all four in-process: Chroma on disk as the
store, an `InMemoryJobQueue` as the queue, a dict standing in for blob storage
and a background task standing in for the worker. None of those is a shortcut
the cloud build could not also take — the client half and the worker half still
meet only through the contract models and the queue, so swapping each stand-in
for the real service changes nothing on either side of it.
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
from core.index.embeddings import embed_texts
from core.index.indexer import run_job
from core.index.job_queue import InMemoryJobQueue, JobQueue
from core.index.vector_store import search_chunks, stored_manifest

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
    """

    def __init__(self, queue: JobQueue | None = None) -> None:
        self._queue = queue or InMemoryJobQueue()
        #: Stands in for Blob Storage: the upload lives here, the queue message
        #: only points at it.
        self._uploads: dict[str, ChunkUpload] = {}
        #: Per-job progress, buffered until the client reads it. `None` marks
        #: the end of a job's stream.
        self._events: dict[str, asyncio.Queue[IndexEvent | None]] = {}
        # asyncio keeps only weak references to tasks; without this set a
        # worker could be garbage-collected mid-job.
        self._workers: set[asyncio.Task] = set()

    # ── Sync ─────────────────────────────────────────────────────────────────

    async def sync(self, request: SyncRequest) -> SyncResult:
        paths = [f.file_path for f in request.files]
        if request.force:
            return SyncResult(changed=paths)

        stored = await asyncio.to_thread(stored_manifest, request.project_id)

        changed = [
            f.file_path
            for f in request.files
            if not (
                f.file_hash
                and f.file_path in stored
                and stored[f.file_path].file_hash == f.file_hash
            )
        ]
        return SyncResult(
            changed=changed,
            removed=sorted(set(stored) - set(paths)),
            unchanged=len(paths) - len(changed),
        )

    # ── Upload and job progress ──────────────────────────────────────────────

    async def upload(self, upload: ChunkUpload) -> str:
        job_id = uuid4().hex
        upload_ref = f"uploads/{job_id}"

        self._uploads[upload_ref] = upload
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
        """The worker: take one message, fetch its upload, run the job."""
        message = await self._queue.receive()
        try:
            # Claim-check: the upload is fetched by reference and then dropped,
            # as the cloud worker deletes the blob once the job has run.
            upload = self._uploads.pop(message.upload_ref)
            async for event in run_job(upload):
                self._publish(message.job_id, event)
        except Exception as exc:  # noqa: BLE001
            # `run_job` already turns failures into events, so this is a defect
            # in the worker itself — still reported, or the client would wait
            # on a stream that never ends.
            logger.exception("Index worker failed outside the job")
            self._publish(message.job_id, IndexEvent(step="error", message=str(exc)))
        finally:
            self._publish(message.job_id, None)

    def _publish(self, job_id: str, event: IndexEvent | None) -> None:
        events = self._events.get(job_id)
        # A client that stopped listening has already dropped its buffer; the
        # job still runs to completion, it just has no audience.
        if events is not None:
            events.put_nowait(event)

    # ── Search ───────────────────────────────────────────────────────────────

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        [vector] = await embed_texts([request.query])
        rows = await asyncio.to_thread(search_chunks, request.project_id, vector, request.n)
        return [SearchHit(**row) for row in rows]
