"""
ChromaDB persistent vector store (Section 3D of architecture spec).

Each project gets its own collection, keyed by a sanitised form of its
absolute path.  Chunks are upserted so re-indexing a project is idempotent.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import chromadb

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
    """Upsert *chunks* with their *embeddings* into the project's collection."""
    if not chunks:
        return
    client = _client()
    col = client.get_or_create_collection(
        _collection_name(project_path),
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[f"{c['file_path']}:{c['start_line']}" for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
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


def search_chunks(
    project_path: str,
    query_embedding: list[float],
    n: int = 10,
) -> list[dict]:
    """Return the *n* most relevant chunks for a given query embedding."""
    client = _client()
    col = client.get_or_create_collection(_collection_name(project_path))
    results = col.query(query_embeddings=[query_embedding], n_results=n)
    return [
        {"content": doc, **meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
