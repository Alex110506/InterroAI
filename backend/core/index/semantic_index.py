"""
SemanticIndex — the runtime's only view of the code index.

Four operations, each shaped like the HTTP call it makes:

  sync(SyncRequest)      → SyncResult      POST /projects/{id}/sync
  upload(ChunkUpload)    → job id          PUT to Blob Storage (SAS) + POST /projects/{id}/jobs
  job_events(job_id)     → IndexEvent …    GET  /jobs/{id}/events  (server-sent events)
  search(SearchRequest)  → [SearchHit]     POST /projects/{id}/search

A search returns *where* the code is — path, line range, and the hash of the
file it was indexed from — never the code itself. The index stores no source
text; the caller reads the lines from its own disk and uses the hash to tell
whether they may have moved since.

`core/remote/semantic_index.py` is the implementation, against the Cloud API.
The worker on the far side assembles the pieces this runtime also uses —
`diff_manifest`, `run_job` — around the cloud adapters in `cloud/adapters/`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from contracts.indexing import (
    ChunkUpload,
    IndexEvent,
    SearchHit,
    SearchRequest,
    SyncRequest,
    SyncResult,
)


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
