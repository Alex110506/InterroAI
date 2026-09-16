"""
A job's progress as server-sent events, and the notifications that make it prompt.

`GET /jobs/{id}/events` streams the job row's event log. Each event goes out
with `id:` set to its position in the log, counting from 1, so a client that
reconnects with `Last-Event-ID: n` resumes after event *n* and misses nothing.
The stream ends once the job has settled and every event has been sent.

Between reads the stream waits for the worker's Postgres NOTIFY
(`cloud.db.jobs.NOTIFY_CHANNEL`). The wait has a timeout, which doubles as the
keep-alive interval. Notifications only make the stream prompt; the stream never
depends on them to be correct. If the listening connection is lost, each wait
runs to its timeout and the stream carries on by polling.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import asyncpg
from sqlalchemy.engine import make_url

from cloud.db.jobs import NOTIFY_CHANNEL, JobRecord

logger = logging.getLogger(__name__)

KEEP_ALIVE = ": keep-alive\n\n"
_TERMINAL_STEPS = ("done", "error")
_CONNECT_TIMEOUT_SECONDS = 5.0
#: After a failed attempt to listen, how long streams poll before trying again.
_RECONNECT_AFTER_SECONDS = 30.0

JobReader = Callable[[], Awaitable[JobRecord | None]]


def asyncpg_dsn(database_url: str) -> str:
    """The app's SQLAlchemy database URL, as a plain DSN asyncpg can connect with."""
    return (
        make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


def sse_event(data: dict, *, event_id: int | None = None) -> str:
    """One server-sent event. JSON without indentation keeps it to a single data line."""
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


class JobNotifications:
    """One LISTEN connection per API process, fanned out to every open stream."""

    def __init__(self, dsn: str | None) -> None:
        #: None listens to nothing, and every stream simply polls.
        self._dsn = dsn
        self._connection: asyncpg.Connection | None = None
        self._lock = asyncio.Lock()
        self._retry_at = 0.0
        self._waiters: dict[str, set[asyncio.Event]] = {}

    @contextlib.asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[asyncio.Event]:
        """An event set whenever *job_id* is notified. Clear it before each read."""
        await self._ensure_listening()
        signal = asyncio.Event()
        self._waiters.setdefault(job_id, set()).add(signal)
        try:
            yield signal
        finally:
            waiters = self._waiters.get(job_id, set())
            waiters.discard(signal)
            if not waiters:
                self._waiters.pop(job_id, None)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    def _listening(self) -> bool:
        return self._connection is not None and not self._connection.is_closed()

    async def _ensure_listening(self) -> None:
        if self._dsn is None or self._listening() or time.monotonic() < self._retry_at:
            return
        async with self._lock:
            if self._listening() or time.monotonic() < self._retry_at:
                return
            try:
                connection = await asyncpg.connect(self._dsn, timeout=_CONNECT_TIMEOUT_SECONDS)
                await connection.add_listener(NOTIFY_CHANNEL, self._on_notify)
            except Exception as exc:  # noqa: BLE001
                # An accelerator, not a dependency: streams still work by polling.
                logger.warning("Job notifications unavailable, streams will poll: %s", exc)
                self._retry_at = time.monotonic() + _RECONNECT_AFTER_SECONDS
                return
            self._connection = connection

    def _on_notify(self, connection, pid, channel, payload: str) -> None:
        for signal in self._waiters.get(payload, ()):
            signal.set()


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
