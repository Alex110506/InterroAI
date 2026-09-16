"""
What every `JobQueue` must do identically — asyncio locally, Service Bus in the cloud.

Subclass `JobQueueContract` and provide:

  * `queue`               an empty queue for this test;
  * `max_delivery_count`  the limit the queue is configured with;
  * `quiet_period`        seconds to wait before concluding no message is
                          coming — an instant in memory, a round trip for
                          Service Bus.

Ordering is deliberately not part of the contract: a Service Bus queue without
sessions does not guarantee it, so nothing may depend on it.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

from contracts.indexing import IndexJobMessage


def _message() -> IndexJobMessage:
    job_id = uuid4().hex
    return IndexJobMessage(job_id=job_id, project_id="project-1", upload_ref=f"uploads/{job_id}")


async def _nothing_arrives(queue, quiet_period: float) -> bool:
    try:
        await asyncio.wait_for(queue.receive(), quiet_period)
    except TimeoutError:
        return True
    return False


class JobQueueContract:
    async def test_a_message_arrives_exactly_as_it_was_sent(self, queue):
        message = _message()
        await queue.enqueue(message)

        delivery = await queue.receive()

        assert delivery.message == message
        await delivery.complete()

    async def test_every_message_sent_is_delivered(self, queue):
        sent = {_message() for _ in range(3)}
        for message in sent:
            await queue.enqueue(message)

        received = set()
        for _ in sent:
            delivery = await queue.receive()
            received.add(delivery.message)
            await delivery.complete()

        assert received == sent

    async def test_a_first_delivery_counts_one(self, queue):
        await queue.enqueue(_message())
        delivery = await queue.receive()
        assert delivery.delivery_count == 1
        await delivery.complete()

    async def test_an_abandoned_message_comes_back_counted_again(self, queue):
        message = _message()
        await queue.enqueue(message)

        first = await queue.receive()
        await first.abandon()
        second = await queue.receive()

        assert second.message == message
        assert second.delivery_count == first.delivery_count + 1
        await second.complete()

    async def test_a_completed_message_never_comes_back(self, queue, quiet_period):
        await queue.enqueue(_message())
        await (await queue.receive()).complete()
        assert await _nothing_arrives(queue, quiet_period)

    async def test_a_dead_lettered_message_never_comes_back(self, queue, quiet_period):
        await queue.enqueue(_message())
        await (await queue.receive()).dead_letter("will never succeed")
        assert await _nothing_arrives(queue, quiet_period)

    async def test_a_message_that_uses_up_its_deliveries_stops_coming_back(
        self, queue, max_delivery_count, quiet_period
    ):
        """A poison message must end in the dead-letter queue, not loop forever."""
        await queue.enqueue(_message())
        for _ in range(max_delivery_count):
            await (await queue.receive()).abandon()
        assert await _nothing_arrives(queue, quiet_period)
