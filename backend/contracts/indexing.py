"""
The data that crosses between the local runtime and the indexing service.

In this build both halves run in one process, but they already meet only
through these models: the runtime never touches the vector store, and the
indexer never touches the filesystem. The cloud build puts HTTP, Service Bus
and Blob Storage between them, and these shapes are what has to survive that
move unchanged — so they are plain data, with no I/O and no imports from the
rest of the backend (`tests/test_boundaries.py` holds that line).

The flow they describe, in order:

  SyncRequest   → SyncResult     which files changed since the index last saw them
  ChunkUpload                    the changed files cut into chunks, plus what to prune
  IndexJobMessage                the queue message: ids and a pointer, never the chunks
  IndexEvent …                   the worker's progress, relayed back to the client
  SearchRequest → [SearchHit]    where matching code lives — never the code itself
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    # Frozen so a message cannot be edited after it was handed across the
    # boundary, and `extra="forbid"` so a misspelt field fails loudly at the
    # edge instead of being silently dropped on the other side.
    model_config = ConfigDict(frozen=True, extra="forbid")


# ── Sync ─────────────────────────────────────────────────────────────────────


class FileState(_Contract):
    """One indexable file as the client currently sees it."""

    #: Relative to the project root, exactly as the walker produced it.
    file_path: str
    #: sha256 of the file's bytes, or "" when it could not be read. An empty
    #: hash never matches a stored one, so an unreadable file is re-attempted.
    file_hash: str


class SyncRequest(_Contract):
    project_id: str
    files: list[FileState] = Field(default_factory=list)
    #: Treat every file as changed. Paired with `ChunkUpload.reset`: the reason
    #: to force is that the index itself is suspect, so it must not be trusted
    #: to say what is unchanged.
    force: bool = False


class SyncResult(_Contract):
    #: Paths to re-chunk and re-embed, in the order the request listed them.
    changed: list[str] = Field(default_factory=list)
    #: Paths the index holds that the client no longer has.
    removed: list[str] = Field(default_factory=list)
    unchanged: int = 0


# ── Upload and job ───────────────────────────────────────────────────────────


class Chunk(_Contract):
    file_path: str
    start_line: int
    end_line: int
    #: Hash of the whole file this chunk was cut from, recorded beside its
    #: vector so the next sync can tell whether the file still matches.
    file_hash: str
    #: Needed to compute the embedding, and deliberately *not* stored with it:
    #: the index keeps vectors, paths and line ranges, never source text.
    content: str


class ChunkUpload(_Contract):
    """
    Everything one index job needs, in one payload.

    `changed_paths` is carried separately from `chunks` because a changed file
    can produce no chunks at all — emptied, or grown past the chunker's size
    cap — and its old chunks still have to be pruned. Deriving the set from
    the chunks alone is how such a file used to keep stale vectors forever.
    """

    project_id: str
    chunks: list[Chunk] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    removed_paths: list[str] = Field(default_factory=list)
    #: Discard everything the index holds for the project before storing.
    reset: bool = False


class IndexJobMessage(_Contract):
    """
    What goes on the queue: identifiers and a pointer to the upload.

    Never the chunks themselves. Queue messages are size-limited, and a large
    repository's upload would not fit — so the upload is stored separately and
    the message only says where (the claim-check pattern).
    """

    job_id: str
    project_id: str
    upload_ref: str


# ── Progress ─────────────────────────────────────────────────────────────────


class IndexEvent(_Contract):
    """
    One progress event from an indexing run.

      {"step": "A"|"B"|"C"|"D", "status": "start"|"progress"|"done", …counts}
      {"step": "done", "embedded", "cached", "skipped", "skipped_files", "deleted", "unchanged"}
      {"step": "error", "message"}

    A–B happen on the client (scan, chunk); C–D and the summary come from the
    worker (embed, prune). Every counter is optional because each step reports
    a different subset of them.
    """

    step: Literal["A", "B", "C", "D", "done", "error"]
    status: Literal["start", "progress", "done"] | None = None

    files: int | None = None
    changed: int | None = None
    removed: int | None = None
    unchanged: int | None = None
    chunks: int | None = None
    total: int | None = None
    embedded: int | None = None
    cached: int | None = None
    skipped: int | None = None
    skipped_files: list[str] | None = None
    stored: int | None = None
    deleted: int | None = None
    message: str | None = None

    def to_wire(self) -> dict:
        """The dict the frontend receives: unset fields omitted, zeros kept."""
        return self.model_dump(exclude_none=True)


# ── Search ───────────────────────────────────────────────────────────────────


class SearchRequest(_Contract):
    project_id: str
    query: str
    n: int = Field(default=5, ge=1, le=10)


class SearchHit(_Contract):
    """
    Where a match lives, and what the file looked like when it was indexed.

    No `content`: the caller reads the lines from its own disk. `file_hash`
    is what lets it notice the file has been edited since, in which case the
    line range may have shifted.
    """

    file_path: str
    start_line: int
    end_line: int
    file_hash: str
    #: Higher is closer. Comparable within one result list, not across indexes.
    score: float
