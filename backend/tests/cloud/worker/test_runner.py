"""
The worker's loop, with in-memory stand-ins for the queue, uploads and job rows.

What is under test is the settlement logic — which outcome completes, abandons
or dead-letters a message — and what the job row ends up saying. Embeddings are
faked, and the store is a throwaway Chroma.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.index.indexer as indexer
from cloud.db.jobs import JobRecord
from cloud.worker.runner import JobRunner
from contracts.indexing import Chunk, ChunkUpload, IndexJobMessage
from core.index.adapters.chroma import ChromaChunkStore, stored_manifest
from core.index.adapters.memory import (
    InMemoryEmbeddingCache,
    InMemoryJobQueue,
    InMemoryUploadStore,
)
from core.index.embeddings import EmbeddedBatch

VECTOR = [0.1, 0.2, 0.3, 0.4]
MAX_DELIVERIES = 3


class FakeJobs:
    """Job rows in a dict, with a switch to make the database 'go away'."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.appends_fail = False

    def add(self, job_id: str, project_id: str, upload_ref: str, status: str = "queued") -> None:
        self.rows[job_id] = {
            "project_id": project_id,
            "upload_ref": upload_ref,
            "status": status,
            "events": [],
            "error": None,
        }

    async def get(self, job_id):
        row = self.rows.get(job_id)
        if row is None:
            return None
        return JobRecord(
            id=job_id,
            project_id=row["project_id"],
            status=row["status"],
            upload_ref=row["upload_ref"],
            events=list(row["events"]),
            error=row["error"],
        )

    async def mark_running(self, job_id):
        self.rows[job_id]["status"] = "running"

    async def append_event(self, job_id, event):
        if self.appends_fail:
            raise ConnectionError("the database went away")
        self.rows[job_id]["events"].append(event.to_wire())
        return len(self.rows[job_id]["events"])

    async def finish(self, job_id, *, error=None):
        self.rows[job_id]["status"] = "failed" if error else "done"
        self.rows[job_id]["error"] = error


@pytest.fixture
def embedded(monkeypatch):
    calls: list[list[str]] = []

    async def fake_batches(texts, **kwargs):
        calls.append(list(texts))
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    return calls


@pytest.fixture
def world(tmp_path, isolated_chroma, embedded):
    jobs = FakeJobs()
    queue = InMemoryJobQueue(max_delivery_count=MAX_DELIVERIES)
    uploads = InMemoryUploadStore()
    runner = JobRunner(
        queue=queue,
        uploads=uploads,
        store=ChromaChunkStore(),
        cache=InMemoryEmbeddingCache(),
        jobs=jobs,
        max_delivery_count=MAX_DELIVERIES,
    )
    return SimpleNamespace(
        jobs=jobs, queue=queue, uploads=uploads, runner=runner, project=str(tmp_path / "proj")
    )


async def _submit(world, job_id="job-1", *, with_upload=True, status="queued") -> None:
    upload_ref = f"uploads/{job_id}"
    if with_upload:
        chunk = Chunk(file_path="a.py", start_line=1, end_line=3, file_hash="h1", content="x = 1")
        await world.uploads.put(
            upload_ref,
            ChunkUpload(project_id=world.project, chunks=[chunk], changed_paths=["a.py"]),
        )
    world.jobs.add(job_id, world.project, upload_ref, status=status)
    await world.queue.enqueue(
        IndexJobMessage(job_id=job_id, project_id=world.project, upload_ref=upload_ref)
    )


async def _handle_next(world) -> None:
    await world.runner.handle(await world.queue.receive())


async def _queue_is_empty(queue) -> bool:
    try:
        await asyncio.wait_for(queue.receive(), 0.05)
    except TimeoutError:
        return True
    return False


# ── Settlement ───────────────────────────────────────────────────────────────


async def test_a_job_runs_to_done_and_its_message_is_completed(world):
    await _submit(world)
    await _handle_next(world)

    row = world.jobs.rows["job-1"]
    assert row["status"] == "done"
    assert "a.py" in stored_manifest(world.project)
    assert len(world.uploads) == 0, "the upload is deleted once its job has run"
    assert await _queue_is_empty(world.queue)
    assert world.queue.dead_letters == []


async def test_every_event_is_recorded_on_the_job_in_order(world):
    await _submit(world)
    await _handle_next(world)

    steps = [event["step"] for event in world.jobs.rows["job-1"]["events"]]
    assert steps == ["C", "C", "C", "D", "D", "done"]


async def test_a_job_that_fails_on_its_data_is_marked_failed_and_not_retried(world, monkeypatch):
    async def refuse(texts, **kwargs):
        raise RuntimeError("provider refused the upload")
        yield  # pragma: no cover

    monkeypatch.setattr(indexer, "embed_batches", refuse)
    await _submit(world)
    await _handle_next(world)

    row = world.jobs.rows["job-1"]
    assert row["status"] == "failed"
    assert "provider refused" in row["error"]
    assert row["events"][-1]["step"] == "error"
    assert await _queue_is_empty(world.queue), "a deterministic failure is not redelivered"


async def test_a_redelivered_job_that_already_finished_is_not_run_again(world, embedded):
    await _submit(world, status="done")
    await _handle_next(world)

    assert embedded == []
    assert await _queue_is_empty(world.queue)


async def test_a_job_whose_worker_died_mid_run_is_run_again(world):
    """Its row still says running; the redelivered message is its second chance."""
    await _submit(world, status="running")
    await _handle_next(world)
    assert world.jobs.rows["job-1"]["status"] == "done"


async def test_a_message_for_no_known_job_is_dead_lettered(world):
    await world.queue.enqueue(IndexJobMessage(job_id="ghost", project_id="p", upload_ref="x"))
    await _handle_next(world)

    assert [reason for _, reason in world.queue.dead_letters] == ["unknown job"]


async def test_a_missing_upload_fails_the_job_and_dead_letters_it(world):
    await _submit(world, with_upload=False)
    await _handle_next(world)

    row = world.jobs.rows["job-1"]
    assert row["status"] == "failed"
    assert row["events"][-1]["step"] == "error", "the client must hear that it is over"
    assert len(world.queue.dead_letters) == 1


# ── Infrastructure failures ──────────────────────────────────────────────────


async def test_an_infrastructure_failure_is_retried_through_redelivery(world):
    await _submit(world)

    world.jobs.appends_fail = True
    await _handle_next(world)
    assert world.jobs.rows["job-1"]["status"] == "running"

    world.jobs.appends_fail = False
    redelivery = await world.queue.receive()
    assert redelivery.delivery_count == 2
    await world.runner.handle(redelivery)

    assert world.jobs.rows["job-1"]["status"] == "done"


async def test_a_job_failing_on_its_last_delivery_is_given_up(world):
    """No client may be left waiting on a job no worker will ever finish."""
    await _submit(world)
    world.jobs.appends_fail = True

    for _ in range(MAX_DELIVERIES):
        await _handle_next(world)

    assert world.jobs.rows["job-1"]["status"] == "failed"
    assert len(world.queue.dead_letters) == 1
    assert await _queue_is_empty(world.queue)


# ── The loop ─────────────────────────────────────────────────────────────────


async def test_run_handles_jobs_until_it_is_stopped(world):
    await _submit(world, "job-1")
    await _submit(world, "job-2")
    stop = asyncio.Event()
    running = asyncio.create_task(world.runner.run(stop))

    for _ in range(200):
        if all(row["status"] == "done" for row in world.jobs.rows.values()):
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(running, 2)

    assert {row["status"] for row in world.jobs.rows.values()} == {"done"}


async def test_an_idle_worker_stops_promptly(world):
    stop = asyncio.Event()
    running = asyncio.create_task(world.runner.run(stop))
    await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(running, 1)


async def test_stopping_mid_job_finishes_the_job_first(world, monkeypatch):
    """Scale-in sends SIGTERM; the job in hand is completed, not abandoned half-way."""
    release = asyncio.Event()

    async def slow_batches(texts, **kwargs):
        await release.wait()
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    monkeypatch.setattr(indexer, "embed_batches", slow_batches)
    await _submit(world)
    stop = asyncio.Event()
    running = asyncio.create_task(world.runner.run(stop))

    for _ in range(200):
        if world.jobs.rows["job-1"]["status"] == "running":
            break
        await asyncio.sleep(0.01)
    stop.set()
    release.set()
    await asyncio.wait_for(running, 2)

    assert world.jobs.rows["job-1"]["status"] == "done"
