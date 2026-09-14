"""
JobQueue — how an index job travels from the runtime to whoever embeds it.

`InMemoryJobQueue` is an `asyncio.Queue` inside this process. The cloud build
replaces it with Service Bus, consumed by the Embed Worker, and two properties
of that system shape this interface even here:

  * **Messages carry identifiers, not data.** An `IndexJobMessage` holds a job
    id and a reference to the uploaded chunks, because queue messages are
    size-limited and a large repository's chunks would not fit.
  * **Delivery is at least once.** The in-memory queue cannot redeliver, but
    the consumer is written as if it might: re-running a job re-upserts the
    same chunk ids and prunes nothing new, so a duplicate costs API time and
    never correctness.
"""
from __future__ import annotations

import asyncio
from typing import Protocol

from contracts.indexing import IndexJobMessage


class JobQueue(Protocol):
    async def enqueue(self, message: IndexJobMessage) -> None: ...

    async def receive(self) -> IndexJobMessage:
        """The next message, waiting for one if the queue is empty."""
        ...


class InMemoryJobQueue:
    """FIFO, in-process. Stands in for Service Bus in the local build."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[IndexJobMessage] = asyncio.Queue()

    async def enqueue(self, message: IndexJobMessage) -> None:
        await self._queue.put(message)

    async def receive(self) -> IndexJobMessage:
        return await self._queue.get()
