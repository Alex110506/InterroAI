"""The in-memory `JobQueue` the index tests run on, held to the shared contract."""
from __future__ import annotations

import pytest
from fakes.queue import InMemoryJobQueue
from port_contracts.job_queue import JobQueueContract

from contracts.indexing import IndexJobMessage


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


async def test_a_delivery_cannot_be_settled_twice():
    """Service Bus refuses it too, so a double settlement is caught here first."""
    queue = InMemoryJobQueue()
    await queue.enqueue(IndexJobMessage(job_id="1", project_id="p", upload_ref="uploads/1"))
    delivery = await queue.receive()
    await delivery.complete()

    with pytest.raises(RuntimeError, match="already settled"):
        await delivery.abandon()
