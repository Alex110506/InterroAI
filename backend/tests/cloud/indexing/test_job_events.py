"""
A job's event stream, frame by frame, against scripted job rows.

The route over HTTP is covered in `test_index_routes.py`, and the notifications
that wake a stream in `test_notifications.py`.
"""
from __future__ import annotations

import asyncio

from cloud.indexing.job_events import KEEP_ALIVE, job_event_stream, sse_event
from cloud.indexing.jobs import JobRecord

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
