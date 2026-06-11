"""
ChromaDB persistent vector store (Section 3D of architecture spec).

Each project gets its own collection, keyed by a sanitised form of its
absolute path.  Chunks are upserted so re-indexing a project is idempotent.
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

_STORE_DIR = Path.home() / ".interroai" / "chroma"


def _collection_name(project_path: str) -> str:
    """Convert an absolute path to a valid ChromaDB collection name (3-63 chars)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", project_path).strip("_-")
    if not safe:
        safe = "project"
    # Must not start with a digit
    if safe[0].isdigit():
        safe = "p_" + safe
    name = safe[:63]
    return name.ljust(3, "_")   # Chroma requires ≥ 3 chars


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
