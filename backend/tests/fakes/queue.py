"""
An in-memory `JobQueue`, held to the same contract as the Service Bus one.

In-process and FIFO, with the parts of Service Bus that only matter when
something goes wrong: an abandoned message comes back counted again, one
abandoned on its last allowed delivery stops coming back, and a delivery
cannot be settled twice.
"""
from __future__ import annotations

import asyncio

from contracts.indexing import IndexJobMessage


class _InMemoryDelivery:
    def __init__(
        self, queue: InMemoryJobQueue, message: IndexJobMessage, delivery_count: int
    ) -> None:
        self.message = message
        self.delivery_count = delivery_count
        self._queue = queue
        self._settled = False

    async def complete(self) -> None:
        self._settle()

    async def abandon(self) -> None:
        self._settle()
        await self._queue._redeliver(self.message, self.delivery_count)

    async def dead_letter(self, reason: str) -> None:
        self._settle()

    def _settle(self) -> None:
        # Service Bus refuses to settle a message twice. So does this, so a
        # double settlement is a bug found by a test, not in Azure.
        if self._settled:
            raise RuntimeError(f"The delivery of job {self.message.job_id} is already settled.")
        self._settled = True


class InMemoryJobQueue:
    """`max_delivery_count` mirrors the Service Bus queue setting of the same name."""

    def __init__(self, *, max_delivery_count: int = 5) -> None:
        self._queue: asyncio.Queue[tuple[IndexJobMessage, int]] = asyncio.Queue()
        self._max_delivery_count = max_delivery_count

    async def enqueue(self, message: IndexJobMessage) -> None:
        await self._queue.put((message, 1))

    async def receive(self) -> _InMemoryDelivery:
        message, delivery_count = await self._queue.get()
        return _InMemoryDelivery(self, message, delivery_count)

    async def _redeliver(self, message: IndexJobMessage, delivery_count: int) -> None:
        if delivery_count < self._max_delivery_count:
            await self._queue.put((message, delivery_count + 1))
