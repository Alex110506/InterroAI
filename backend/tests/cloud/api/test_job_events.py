"""
A job's event stream, frame by frame, and the notifications that wake it.

The stream is tested against scripted job rows; the LISTEN connection against
real Postgres (`pytest -m integration`). The route over HTTP is covered in
`test_index_routes.py`.
"""
from __future__ import annotations

import asyncio

import pytest

from cloud.api.job_events import (
    KEEP_ALIVE,
    JobNotifications,
    asyncpg_dsn,
    job_event_stream,
    sse_event,
)
from cloud.db.jobs import JobRecord, JobRepository
from cloud.db.session import service_scope
from contracts.indexing import IndexEvent

STARTED = {"step": "C", "status": "start", "total": 1}
DONE = {"step": "done", "embedded": 1}


def _job(events: list[dict], status: str = "running", error: str | None = None) -> JobRecord:
    return JobRecord(
        id="job-1", project_id="p", status=status, upload_ref="u", events=events, error=error
    )


class ScriptedReader:
    """Returns each snapshot in turn, then keeps returning the last one."""

    def __init__(self, *snapshots: JobRecord | None) -> None:
        self._snapshots = list(snapshots)

    async def __call__(self) -> JobRecord | None:
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]


async def _frames(reader, *, after=0, heartbeat=5.0, signal=None) -> list[str]:
    stream = job_event_stream(
        reader, signal or asyncio.Event(), after=after, heartbeat_seconds=heartbeat
    )
    return [frame async for frame in stream]


def test_an_event_is_one_data_line_with_its_position_as_id():
    frame = sse_event({"step": "error", "message": "two\nlines"}, event_id=3)
    assert frame == 'id: 3\ndata: {"step":"error","message":"two\\nlines"}\n\n'


async def test_a_settled_job_streams_its_whole_log_then_ends():
    frames = await _frames(ScriptedReader(_job([STARTED, DONE], status="done")))
    assert frames == [sse_event(STARTED, event_id=1), sse_event(DONE, event_id=2)]


async def test_a_reconnecting_client_resumes_after_its_last_event():
    frames = await _frames(ScriptedReader(_job([STARTED, DONE], status="done")), after=1)
    assert frames == [sse_event(DONE, event_id=2)]


async def test_new_events_go_out_as_soon_as_the_worker_signals():
    signal = asyncio.Event()
    reader = ScriptedReader(_job([STARTED]), _job([STARTED, DONE], status="done"))
    stream = job_event_stream(reader, signal, after=0, heartbeat_seconds=60)

    assert await anext(stream) == sse_event(STARTED, event_id=1)
    waiting = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0.05)
    assert not waiting.done(), "nothing new yet: the stream must wait, not spin"

    signal.set()

    assert await asyncio.wait_for(waiting, 1) == sse_event(DONE, event_id=2)
    assert [frame async for frame in stream] == []


async def test_an_idle_stream_sends_keep_alives_and_keeps_checking():
    reader = ScriptedReader(_job([]), _job([]), _job([DONE], status="done"))
    frames = await _frames(reader, heartbeat=0.01)
    assert frames == [KEEP_ALIVE, KEEP_ALIVE, sse_event(DONE, event_id=1)]


async def test_a_job_settled_without_a_final_event_still_tells_the_client():
    frames = await _frames(ScriptedReader(_job([STARTED], status="failed", error="lost lock")))
    assert frames[-1] == sse_event({"step": "error", "message": "lost lock"})


async def test_a_job_that_disappears_ends_the_stream():
    assert await _frames(ScriptedReader(None)) == []


async def test_without_a_database_a_subscription_only_times_out():
    notifications = JobNotifications(None)
    async with notifications.subscribe("job-1") as signal:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(signal.wait(), 0.01)


def test_the_database_url_becomes_an_asyncpg_dsn():
    dsn = asyncpg_dsn("postgresql+asyncpg://app:s3cret@db.example:5432/interroai")
    assert dsn == "postgresql://app:s3cret@db.example:5432/interroai"


@pytest.mark.integration
async def test_a_notify_wakes_the_subscribers_of_that_job_only(
    app_sessions, database_urls, pg_new_project
):
    jobs = JobRepository(lambda: service_scope(app_sessions))
    mine = await jobs.create(await pg_new_project(), "uploads/mine")
    theirs = await jobs.create(await pg_new_project(), "uploads/theirs")
    notifications = JobNotifications(asyncpg_dsn(database_urls.app))
    try:
        async with (
            notifications.subscribe(mine) as my_signal,
            notifications.subscribe(theirs) as their_signal,
        ):
            await jobs.append_event(mine, IndexEvent(step="C", status="start", total=1))

            await asyncio.wait_for(my_signal.wait(), 5)
            assert not their_signal.is_set()
    finally:
        await notifications.close()
