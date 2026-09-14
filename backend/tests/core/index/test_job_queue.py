"""The in-memory stand-in for Service Bus."""
from __future__ import annotations

import asyncio

from contracts.indexing import IndexJobMessage
from core.index.job_queue import InMemoryJobQueue


def _message(job_id: str) -> IndexJobMessage:
    return IndexJobMessage(job_id=job_id, project_id="p", upload_ref=f"uploads/{job_id}")


async def test_messages_come_out_in_the_order_they_went_in():
    queue = InMemoryJobQueue()
    for job_id in ("1", "2", "3"):
        await queue.enqueue(_message(job_id))

    assert [(await queue.receive()).job_id for _ in range(3)] == ["1", "2", "3"]


async def test_receiving_from_an_empty_queue_waits_rather_than_fails():
    queue = InMemoryJobQueue()
    waiting = asyncio.create_task(queue.receive())
    await asyncio.sleep(0)
    assert not waiting.done()

    await queue.enqueue(_message("late"))
    assert (await asyncio.wait_for(waiting, timeout=1)).job_id == "late"


async def test_a_message_comes_out_exactly_as_it_went_in():
    queue = InMemoryJobQueue()
    message = _message("1")
    await queue.enqueue(message)
    assert await queue.receive() == message
