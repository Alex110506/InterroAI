"""
The interfaces behind the index: what indexing needs, not who provides it.

`indexer.py`, `embeddings.py` and `semantic_index.py` are written against these,
and each has a local implementation in `core/index/adapters/` and a cloud one
in `cloud/adapters/`:

  ChunkStore       vectors and locations           Chroma on disk     │ Postgres + pgvector
  EmbeddingCache   vectors by content hash         Redis (optional)   │ a Postgres table
  UploadStore      one job's chunks, by reference  a dict             │ Azure Blob Storage
  JobQueue         index jobs, acknowledged        asyncio.Queue      │ Azure Service Bus

What every implementation must do the same way is pinned by the suites in
`tests/port_contracts/`, which run against each of them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from contracts.indexing import ChunkUpload, IndexJobMessage, SearchHit


def chunk_id(file_path: str, start_line: int) -> str:
    """
    The stable id of a chunk: its file and the line it starts at.

    Deterministic on purpose — re-indexing an unchanged file overwrites its own
    chunks instead of accumulating duplicates. The flip side is that ids a file
    no longer produces have to be deleted explicitly.
    """
    return f"{file_path}:{start_line}"


def split_chunk_id(key: str) -> tuple[str, int]:
    """The inverse of `chunk_id`. A path may contain ':'; a line number never does."""
    file_path, _, start_line = key.rpartition(":")
    return file_path, int(start_line)


def content_digest(text: str) -> str:
    """
    The key an embedding is cached under: a hash of the text alone.

    Content addressing is what makes a renamed or moved file free to re-index —
    its text is unchanged, so its vectors already exist.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class StoredFile:
    """What a store currently holds for one file."""

    file_hash: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk ready to store: where it is, its file's hash and its vector — never its text."""

    file_path: str
    start_line: int
    end_line: int
    file_hash: str
    vector: list[float]

    @property
    def id(self) -> str:
        return chunk_id(self.file_path, self.start_line)


class UnusableUploadError(Exception):
    """An upload that can never be indexed, however often the job is retried."""


class UploadNotFoundError(UnusableUploadError, LookupError):
    """Nothing is stored under this reference: never uploaded, or already consumed."""


class ChunkStore(Protocol):
    async def manifest(self, project_id: str) -> dict[str, StoredFile]:
        """Everything indexed for a project, grouped by file. Empty for an unknown one."""
        ...

    async def apply(
        self,
        project_id: str,
        *,
        upserts: list[IndexedChunk],
        delete: list[str],
        reset: bool = False,
    ) -> int:
        """
        One job's whole change to the index, in a single call.

        Discards everything first when *reset*, removes the *delete* ids and
        upserts *upserts*. Returns how many of the *delete* ids were removed.
        Atomic where the store can be (Postgres: one transaction), so a search
        sees the index from before the job or after it, never half of each.
        """
        ...

    async def search(self, project_id: str, vector: list[float], n: int) -> list[SearchHit]:
        """The *n* nearest chunks, best first. Empty for an unknown project."""
        ...


class EmbeddingCache(Protocol):
    async def get(self, model: str, digests: list[str]) -> dict[str, list[float]]:
        """The cached vectors for *digests* under *model*; misses are simply absent."""
        ...

    async def put(self, model: str, vectors: dict[str, list[float]]) -> None:
        """Remember *vectors* by digest. Writing an entry twice is harmless."""
        ...


class UploadStore(Protocol):
    async def put(self, upload_ref: str, upload: ChunkUpload) -> None: ...

    async def get(self, upload_ref: str) -> ChunkUpload:
        """Raises `UploadNotFoundError` when nothing is stored under *upload_ref*."""
        ...

    async def delete(self, upload_ref: str) -> None:
        """Deleting an upload that is already gone is not an error."""
        ...


class Delivery(Protocol):
    """
    One received message, locked to its consumer until it is settled.

    Exactly one of the three settlements should be called. A consumer that dies
    without settling loses the lock and the queue delivers the message again —
    that, not a retry loop in code, is how a crashed job gets its second run.
    """

    message: IndexJobMessage
    #: 1 on the first delivery, one higher on each redelivery.
    delivery_count: int

    async def complete(self) -> None:
        """Handled: never deliver it again."""
        ...

    async def abandon(self) -> None:
        """Not handled: deliver it again, unless it has used up its deliveries."""
        ...

    async def dead_letter(self, reason: str) -> None:
        """Will never succeed: set it aside for inspection instead of redelivering it."""
        ...


class JobQueue(Protocol):
    async def enqueue(self, message: IndexJobMessage) -> None: ...

    async def receive(self) -> Delivery:
        """Wait for the next message and lock it to this consumer."""
        ...
