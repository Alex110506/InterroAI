"""
The worker half of indexing: embed an upload, store it, prune what it replaced.

This is the code that becomes the Embed Worker in the cloud build. It receives
one `ChunkUpload` — chunks the runtime already cut and hashed — and knows
nothing about where they came from: it opens no project files and reads no
`.gitignore`. That is the property that lets it run on a machine that has never
seen the repository, and `tests/test_boundaries.py` keeps it that way.

It yields the C (embedding) and D (pruning) steps and the final summary. A
failure becomes an `error` event rather than an exception, because a worker has
no caller to raise into — it records the outcome on the job, and the client
reads it from there.

The rules carried over from the single-process indexer:

  * **Stored batch by batch.** A job that dies on batch 59 of 60 keeps the 58
    already paid for, and the next sync sees the rest as still changed.
  * **Pruned after storing, not before.** A job that dies part-way leaves the
    old vectors in place rather than a hole.
  * **Upserting is not enough.** Chunk ids are `file_path:start_line`, so a
    removed file, and the chunks past the new end of a file that shrank, would
    otherwise answer searches forever.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from contracts.indexing import ChunkUpload, IndexEvent
from core.errors import InterroAIError
from core.index.embeddings import embed_batches
from core.index.vector_store import (
    StoredFile,
    chunk_id,
    delete_ids,
    reset_collection,
    store_chunks,
    stored_manifest,
)

logger = logging.getLogger(__name__)


async def run_job(upload: ChunkUpload) -> AsyncIterator[IndexEvent]:
    """Process one upload, yielding progress. Never raises; failures are events."""
    try:
        async for event in _job_steps(upload):
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
    rows: list[dict],
) -> list[str]:
    """
    Ids this job makes obsolete: every chunk of a removed file, and every chunk
    a changed file used to have that its new chunks do not reproduce.

    Decided from what the store holds when the job runs, not from what the
    client believed at sync time. The worker is the only writer, and in the
    cloud the two moments can be minutes apart.
    """
    fresh = {chunk_id(row) for row in rows}
    held = {path: stored[path].ids for path in stored}
    stale = [key for path in upload.removed_paths for key in held.get(path, ())]
    stale += [
        key
        for path in upload.changed_paths
        for key in held.get(path, ())
        if key not in fresh
    ]
    return stale


async def _job_steps(upload: ChunkUpload) -> AsyncIterator[IndexEvent]:
    project = upload.project_id

    if upload.reset:
        # The store itself is what the user is discarding, so there is nothing
        # in it worth reconciling against.
        await asyncio.to_thread(reset_collection, project)
        stored: dict[str, StoredFile] = {}
    else:
        stored = await asyncio.to_thread(stored_manifest, project)

    rows = [chunk.model_dump() for chunk in upload.chunks]
    stale = _stale_ids(upload, stored, rows)

    # ── C: embed, persisted batch by batch ───────────────────────────────────
    stored_count = 0
    cached_count = 0
    skipped: list[str] = []

    if rows:
        yield IndexEvent(step="C", status="start", total=len(rows))

        async for batch in embed_batches([row["content"] for row in rows]):
            if batch.indices:
                batch_rows = [rows[i] for i in batch.indices]
                await asyncio.to_thread(store_chunks, project, batch_rows, batch.vectors)
                stored_count += len(batch_rows)
                cached_count += batch.from_cache

            for position in batch.failures:
                # Its file's other chunks still store the current hash, so the
                # next sync sees the file as unchanged and does not re-attempt
                # this one. Deliberate: an item error is deterministic (over the
                # token limit, refused content), so retrying on every run would
                # re-pay for the chunks that *do* work to fail identically on
                # this one. The summary reports it; a forced rebuild retries it.
                skipped.append(rows[position]["file_path"])

            yield IndexEvent(
                step="C",
                status="progress",
                embedded=stored_count,
                total=len(rows),
                cached=cached_count,
                skipped=len(skipped),
            )

        yield IndexEvent(
            step="C", status="done", total=len(rows), cached=cached_count, skipped=len(skipped)
        )

    # ── D: prune what the new chunks replaced or outlived ────────────────────
    yield IndexEvent(step="D", status="start")
    deleted = await asyncio.to_thread(delete_ids, project, stale)
    yield IndexEvent(step="D", status="done", stored=stored_count, deleted=deleted)

    yield IndexEvent(
        step="done",
        embedded=stored_count,
        cached=cached_count,
        skipped=len(skipped),
        skipped_files=sorted(set(skipped)),
        deleted=deleted,
    )
