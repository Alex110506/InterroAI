"""
A job's progress as server-sent events.

`GET /jobs/{id}/events` streams the job row's event log. Each event goes out
with `id:` set to its position in the log, counting from 1, so a client that
reconnects with `Last-Event-ID: n` resumes after event *n* and misses nothing.
The stream ends once the job has settled and every event has been sent.

Between reads the stream waits on a signal from
`cloud.db.notifications.JobNotifications`, set when the worker's NOTIFY for this
job arrives. The wait has a timeout, which doubles as the keep-alive interval,
so a stream whose notifications never come still finishes — by polling.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from cloud.db.jobs import JobRecord

KEEP_ALIVE = ": keep-alive\n\n"
_TERMINAL_STEPS = ("done", "error")

JobReader = Callable[[], Awaitable[JobRecord | None]]


def sse_event(data: dict, *, event_id: int | None = None) -> str:
    """One server-sent event. JSON without indentation keeps it to a single data line."""
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


async def job_event_stream(
    read_job: JobReader,
    signal: asyncio.Event,
    *,
    after: int,
    heartbeat_seconds: float,
) -> AsyncIterator[str]:
    """
    The frames of one job's event stream, from event *after* + 1 until the job settles.

    *signal* is cleared before every read, so a notification that arrives while
    the row is being read still wakes the next wait instead of being lost.
    """
    sent = after
    while True:
        signal.clear()
        job = await read_job()
        if job is None:
            # Gone: its project was deleted mid-stream.
            return

        for position, event in enumerate(job.events[sent:], start=sent + 1):
            yield sse_event(event, event_id=position)
        sent = max(sent, len(job.events))

        if job.settled:
            if not job.events or job.events[-1].get("step") not in _TERMINAL_STEPS:
                # Settled without a closing event, because its last write failed.
                # The client must still learn that the job is over.
                message = job.error or "The index job ended without reporting why."
                yield sse_event({"step": "error", "message": message})
            return

        try:
            await asyncio.wait_for(signal.wait(), heartbeat_seconds)
        except TimeoutError:
            yield KEEP_ALIVE
