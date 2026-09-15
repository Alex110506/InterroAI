"""
The worker half of indexing: embed one upload, then change the index in one step.

This is the code that becomes the Embed Worker in the cloud build. It receives
one `ChunkUpload` — chunks the runtime already cut and hashed — and knows
nothing about where they came from: it opens no project files and reads no
`.gitignore`. That is what lets it run on a machine that has never seen the
repository, and `tests/test_boundaries.py` keeps it that way. The store and the
cache are handed in as ports (`core/index/ports.py`), so the same code runs
over Chroma on a laptop and pgvector in Azure.

It yields the C (embedding) and D (writing) steps and the final summary. A
failure becomes an `error` event rather than an exception, because a worker has
no caller to raise into — it records the outcome on the job, and the client
reads it from there.

The rules:

  * **The index changes once, at the end, atomically.** Every chunk is embedded
    first; then a single `ChunkStore.apply` upserts the new chunks, deletes the
    stale ones and, for a forced rebuild, resets the project. A search sees the
    index from before the job or after it, never half of each. Money already
    spent still survives a failure: each batch lands in the `EmbeddingCache` as
    it completes, so a retried job pays only for what never succeeded.
  * **Upserting is not enough.** Chunk ids are `file_path:start_line`, so a
    removed file, and the chunks past the new end of a file that shrank, would
    otherwise answer searches forever.
  * **Staleness is decided when the job runs,** from what the store holds then,
    not from what the client believed at sync time — in the cloud the two
    moments can be minutes apart.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from contracts.indexing import ChunkUpload, IndexEvent
from core.errors import InterroAIError
from core.index.embeddings import embed_batches
from core.index.ports import ChunkStore, EmbeddingCache, IndexedChunk, StoredFile

logger = logging.getLogger(__name__)

_NOTHING_STORED = StoredFile(file_hash="", ids=())


async def run_job(
    upload: ChunkUpload,
    *,
    store: ChunkStore,
    cache: EmbeddingCache | None = None,
) -> AsyncIterator[IndexEvent]:
    """Process one upload, yielding progress. Never raises; failures are events."""
    try:
        async for event in _job_steps(upload, store, cache):
            yield event
    except InterroAIError as exc:
        # Expected (no API key, say) and already phrased for a person.
        logger.info("Index job stopped: %s", exc)
        yield IndexEvent(step="error", message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in an index job")
        yield IndexEvent(step="error", message=str(exc))


def _stale_ids(
    upload: ChunkUpload,
    stored: dict[str, StoredFile],
    fresh: set[str],
) -> list[str]:
    """
    Ids this job makes obsolete: every chunk of a removed file, and every chunk
    a changed file used to have that this job is not writing again.

    "Not writing again" includes a chunk whose new version could not be
    embedded: its old vector describes text the file no longer has.
    """
    stale = [
        key for path in upload.removed_paths for key in stored.get(path, _NOTHING_STORED).ids
    ]
    stale += [
        key
        for path in upload.changed_paths
        for key in stored.get(path, _NOTHING_STORED).ids
        if key not in fresh
    ]
    return stale


async def _job_steps(
    upload: ChunkUpload,
    store: ChunkStore,
    cache: EmbeddingCache | None,
) -> AsyncIterator[IndexEvent]:
    project = upload.project_id
    # A forced rebuild discards the store, so nothing in it is worth reconciling against.
    stored = {} if upload.reset else await store.manifest(project)
    chunks = upload.chunks

    # ── C: embed everything before anything is written ───────────────────────
    vectors: dict[int, list[float]] = {}
    cached = 0
    skipped: list[str] = []

    if chunks:
        yield IndexEvent(step="C", status="start", total=len(chunks))

        async for batch in embed_batches([chunk.content for chunk in chunks], cache=cache):
            vectors.update(zip(batch.indices, batch.vectors, strict=True))
            cached += batch.from_cache
            for position in batch.failures:
                # Deliberate: an item error is deterministic (over the token
                # limit, refused content). The file's other chunks record its
                # current hash, so the next sync does not retry this one and
                # re-pay for the rest. The summary reports it; a forced rebuild
                # retries it.
                skipped.append(chunks[position].file_path)

            yield IndexEvent(
                step="C",
                status="progress",
                embedded=len(vectors),
                total=len(chunks),
                cached=cached,
                skipped=len(skipped),
            )

        yield IndexEvent(
            step="C", status="done", total=len(chunks), cached=cached, skipped=len(skipped)
        )

    # ── D: one write — upsert, prune and reset together ──────────────────────
    upserts = [
        IndexedChunk(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            file_hash=chunk.file_hash,
            vector=vectors[position],
        )
        for position, chunk in enumerate(chunks)
        if position in vectors
    ]
    stale = _stale_ids(upload, stored, fresh={chunk.id for chunk in upserts})

    yield IndexEvent(step="D", status="start")
    deleted = await store.apply(project, upserts=upserts, delete=stale, reset=upload.reset)
    yield IndexEvent(step="D", status="done", stored=len(upserts), deleted=deleted)

    yield IndexEvent(
        step="done",
        embedded=len(upserts),
        cached=cached,
        skipped=len(skipped),
        skipped_files=sorted(set(skipped)),
        deleted=deleted,
    )
