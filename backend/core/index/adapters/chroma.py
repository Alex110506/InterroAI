"""
The local `ChunkStore`: ChromaDB, persistent on disk (Section 3D of architecture spec).

Each project gets its own collection, keyed by a sanitised form of its
absolute path. Chunks are upserted so re-indexing a project is idempotent.

Upserting alone is not enough, though: an id is `file_path:start_line`, so
deleting or renaming a file leaves its old chunks behind forever, and a file
that *shrank* keeps the chunks past its new end. Those orphans still answer
searches, attributing content to files that no longer contain it. So the store
also has to be able to say what it currently holds (`stored_manifest`) and to
drop specific ids (`delete_ids`).

That manifest doubles as the input to incremental indexing: each chunk carries
the hash of the file it came from, so the indexer can tell which files actually
changed. It lives here, beside the vectors it describes, rather than in the
Redis cache — a manifest that can be evicted independently of the vectors would
eventually disagree with them, and reconciling against a wrong manifest is how
orphans come back.

**No source text is stored.** A chunk's content is what gets embedded, but only
its vector, location and file hash are kept. A search therefore answers with
*where* the code is, and the runtime reads the lines from its own disk — the
same arrangement the cloud index uses, where keeping users' source at rest is a
liability with nothing to show for it.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import chromadb

from contracts.indexing import SearchHit
from core.index.ports import IndexedChunk, StoredFile, chunk_id

logger = logging.getLogger(__name__)

_STORE_DIR = Path.home() / ".interroai" / "chroma"


def _collection_name(project_path: str) -> str:
    """
    Convert an absolute path to a valid ChromaDB collection name.

    Chroma requires 3-63 characters from [a-zA-Z0-9._-] that both *start and
    end* with an alphanumeric. The end-anchor is easy to miss: padding a short
    name with "_" or truncating at 63 characters mid-separator both produce a
    trailing "_" that Chroma rejects outright.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", project_path).strip("_-")
    if not safe:
        safe = "project"
    # Must not start with a digit
    if safe[0].isdigit():
        safe = "p_" + safe
    # Truncation can land on a separator, so re-strip afterwards.
    name = safe[:63].rstrip("_-")
    # Pad with an alphanumeric, never "_", so the end-anchor still holds.
    return name.ljust(3, "0")


def _client() -> chromadb.ClientAPI:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_STORE_DIR))


def store_chunks(
    project_path: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert *chunks* with their *embeddings* into the project's collection.

    Each chunk's `content` is ignored here on purpose: it has already served
    its purpose (being embedded), and the store keeps no source text.
    """
    if not chunks:
        return
    client = _client()
    col = client.get_or_create_collection(
        _collection_name(project_path),
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[chunk_id(c["file_path"], c["start_line"]) for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                # Written by the indexer so a later run can tell whether the
                # file behind this chunk has changed. Empty when the caller
                # does not track it (a one-off store, or an older index).
                "file_hash": c.get("file_hash", ""),
            }
            for c in chunks
        ],
    )


def collection_size(project_path: str) -> int:
    """
    Number of chunks already stored for *project_path*.

    Lets a caller skip a re-embed it would only pay for again: chunks are
    upserted, so re-indexing an unchanged project produces identical vectors
    at full API cost. Returns 0 when nothing has been indexed yet.
    """
    try:
        col = _client().get_or_create_collection(_collection_name(project_path))
        return col.count()
    except Exception:
        # A corrupt or unreadable store should send the caller down the
        # "needs indexing" path, not crash it.
        logger.warning("Could not read the vector store for %r.", project_path, exc_info=True)
        return 0


def stored_manifest(project_path: str) -> dict[str, StoredFile]:
    """
    What is currently indexed, grouped by file path.

    One read of the whole collection's metadata — the ids and hashes only, not
    the vectors — which is what lets the indexer decide, locally and for free,
    which files need re-embedding and which stored chunks are now orphans.

    An unreadable store reports as empty, sending the caller down the "index
    everything" path rather than crashing it.
    """
    hashes: dict[str, str] = {}
    ids: dict[str, list[str]] = {}
    try:
        col = _client().get_or_create_collection(_collection_name(project_path))
        stored = col.get(include=["metadatas"])
        # Parallel lists, one metadata per id — a mismatch means the response
        # cannot be trusted, which is handled below as an unreadable store.
        paired = zip(stored.get("ids") or [], stored.get("metadatas") or [], strict=True)
        for chunk_key, meta in paired:
            # A chunk with no file to attribute it to can never be refreshed by
            # any run; reported under "" so the caller prunes it.
            file_path = (meta or {}).get("file_path") or ""
            ids.setdefault(file_path, []).append(chunk_key)
            # Chunks of one file are written in a single upsert and therefore
            # share a hash; first one wins.
            hashes.setdefault(file_path, (meta or {}).get("file_hash") or "")
    except Exception:
        logger.warning("Could not read the vector store for %r.", project_path, exc_info=True)
        return {}

    return {
        path: StoredFile(file_hash=hashes[path], ids=tuple(chunk_ids))
        for path, chunk_ids in ids.items()
    }


def delete_ids(project_path: str, ids: list[str]) -> int:
    """Drop specific chunks. Returns how many ids were handed over."""
    if not ids:
        return 0
    try:
        col = _client().get_or_create_collection(_collection_name(project_path))
        col.delete(ids=list(ids))
    except Exception:
        # Leaving an orphan in place is bad but survivable, and the next run
        # will try again; taking down an otherwise successful index over it
        # would be worse.
        logger.warning(
            "Could not prune %d stale chunk(s) for %r.", len(ids), project_path, exc_info=True
        )
        return 0
    return len(ids)


def reset_collection(project_path: str) -> None:
    """
    Throw the project's collection away entirely.

    Backs an explicit `--reindex` / `/index`: incremental reconciliation is the
    normal path, so the reason to ask for a full rebuild is that the store
    itself is suspect — a corrupt collection, or vectors from a different
    embedding model. Reconciling against a suspect store would preserve exactly
    what the user is trying to discard.
    """
    try:
        _client().delete_collection(_collection_name(project_path))
    except Exception:
        # Most commonly: there is no collection yet. Nothing to reset.
        logger.debug("No collection to reset for %r.", project_path, exc_info=True)


def search_chunks(
    project_path: str,
    query_embedding: list[float],
    n: int = 10,
) -> list[dict]:
    """
    The *n* chunks nearest to *query_embedding*, best first.

    Each result is a location — `file_path`, `start_line`, `end_line` — plus
    the `file_hash` the chunk was indexed from and a similarity `score`. There
    is no text, because the store never had it: the caller reads the lines from
    disk, and compares the hash to tell whether they may have moved since.
    """
    client = _client()
    col = client.get_or_create_collection(_collection_name(project_path))
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["metadatas", "distances"],
    )
    return [
        {
            "file_path": meta["file_path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "file_hash": meta.get("file_hash") or "",
            # Cosine distance turned into similarity, so "higher is closer".
            "score": 1.0 - distance,
        }
        for meta, distance in zip(results["metadatas"][0], results["distances"][0], strict=True)
    ]


class ChromaChunkStore:
    """
    `ChunkStore` over the functions above, run off the event loop.

    Chroma has no transactions, so `apply` is ordered to fail safe rather than
    being atomic: reset, then upsert, then delete. A crash between the upsert
    and the delete leaves stale chunks, which the next sync prunes — it never
    leaves one missing.
    """

    async def manifest(self, project_id: str) -> dict[str, StoredFile]:
        return await asyncio.to_thread(stored_manifest, project_id)

    async def apply(
        self,
        project_id: str,
        *,
        upserts: list[IndexedChunk],
        delete: list[str],
        reset: bool = False,
    ) -> int:
        def write() -> int:
            if reset:
                reset_collection(project_id)
            if upserts:
                store_chunks(
                    project_id,
                    [_metadata_row(chunk) for chunk in upserts],
                    [chunk.vector for chunk in upserts],
                )
            return delete_ids(project_id, delete)

        return await asyncio.to_thread(write)

    async def search(self, project_id: str, vector: list[float], n: int) -> list[SearchHit]:
        rows = await asyncio.to_thread(search_chunks, project_id, vector, n)
        return [SearchHit(**row) for row in rows]


def _metadata_row(chunk: IndexedChunk) -> dict:
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "file_hash": chunk.file_hash,
    }
