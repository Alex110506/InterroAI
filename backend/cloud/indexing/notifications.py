"""
The API's ear for the worker's NOTIFYs: one LISTEN connection per process.

`cloud.indexing.jobs.JobRepository.append_event` sends a NOTIFY on `NOTIFY_CHANNEL`,
carrying the job id, every time the worker records an event. `JobNotifications`
listens on one dedicated connection and fans each one out to the streams
waiting on that job (`cloud/indexing/job_events.py`).

Notifications only make a stream prompt; no stream depends on them to be
correct. If the connection cannot be opened, or is lost, every wait simply runs
to its timeout and the streams carry on by polling.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator

import asyncpg
from sqlalchemy.engine import make_url

from cloud.indexing.jobs import NOTIFY_CHANNEL

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SECONDS = 5.0
#: After a failed attempt to listen, how long streams poll before trying again.
_RECONNECT_AFTER_SECONDS = 30.0


def asyncpg_dsn(database_url: str) -> str:
    """The app's SQLAlchemy database URL, as a plain DSN asyncpg can connect with."""
    return (
        make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


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
