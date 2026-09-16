"""The in-memory stand-ins for Blob Storage, Service Bus and the embedding cache."""
from __future__ import annotations

import asyncio

import pytest
from port_contracts.embedding_cache import EmbeddingCacheContract
from port_contracts.job_queue import JobQueueContract
from port_contracts.upload_store import UploadStoreContract

from contracts.indexing import IndexJobMessage
from core.index.adapters.memory import (
    InMemoryEmbeddingCache,
    InMemoryJobQueue,
    InMemoryUploadStore,
)

# ── The contracts ────────────────────────────────────────────────────────────


class TestInMemoryUploadStore(UploadStoreContract):
    @pytest.fixture
    def uploads(self):
        return InMemoryUploadStore()


class TestInMemoryEmbeddingCache(EmbeddingCacheContract):
    @pytest.fixture
    def cache(self):
        return InMemoryEmbeddingCache()

    @pytest.fixture
    def dimensions(self):
        return 8


class TestInMemoryJobQueue(JobQueueContract):
    @pytest.fixture
    def max_delivery_count(self):
        return 3

    @pytest.fixture
    def queue(self, max_delivery_count):
        return InMemoryJobQueue(max_delivery_count=max_delivery_count)

    @pytest.fixture
    def quiet_period(self):
        return 0.05


# ── What only the in-memory queue promises ───────────────────────────────────


def _message(job_id: str) -> IndexJobMessage:
    return IndexJobMessage(job_id=job_id, project_id="p", upload_ref=f"uploads/{job_id}")


async def test_messages_come_out_in_the_order_they_went_in():
    queue = InMemoryJobQueue()
    for job_id in ("1", "2", "3"):
        await queue.enqueue(_message(job_id))

    received = []
    for _ in range(3):
        delivery = await queue.receive()
        received.append(delivery.message.job_id)
        await delivery.complete()

    assert received == ["1", "2", "3"]


async def test_receiving_from_an_empty_queue_waits_rather_than_fails():
    queue = InMemoryJobQueue()
    waiting = asyncio.create_task(queue.receive())
    await asyncio.sleep(0)
    assert not waiting.done()

    await queue.enqueue(_message("late"))
    assert (await asyncio.wait_for(waiting, timeout=1)).message.job_id == "late"


async def test_a_delivery_cannot_be_settled_twice():
    """Service Bus refuses it too, so a double settlement is caught here first."""
    queue = InMemoryJobQueue()
    await queue.enqueue(_message("1"))
    delivery = await queue.receive()
    await delivery.complete()

    with pytest.raises(RuntimeError, match="already settled"):
        await delivery.abandon()


async def test_a_dead_letter_keeps_its_reason():
    queue = InMemoryJobQueue()
    message = _message("1")
    await queue.enqueue(message)

    await (await queue.receive()).dead_letter("the upload was never written")

    assert queue.dead_letters == [(message, "the upload was never written")]


async def test_a_message_out_of_deliveries_is_dead_lettered_not_dropped():
    queue = InMemoryJobQueue(max_delivery_count=2)
    message = _message("1")
    await queue.enqueue(message)

    await (await queue.receive()).abandon()
    await (await queue.receive()).abandon()

    assert queue.dead_letters == [(message, "MaxDeliveryCountExceeded")]


async def test_an_empty_upload_store_is_empty():
    uploads = InMemoryUploadStore()
    assert len(uploads) == 0
