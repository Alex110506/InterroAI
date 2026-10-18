"""
The cloud JobQueue on the Service Bus emulator. Needs the local stack:
`pytest -m integration`. Slower than the rest — "nothing arrived" can only be
concluded by waiting.
"""
from __future__ import annotations

import pytest
from port_contracts.job_queue import JobQueueContract

pytestmark = pytest.mark.integration


class TestServiceBusJobQueue(JobQueueContract):
    @pytest.fixture
    def queue(self, service_bus_queue):
        return service_bus_queue

    @pytest.fixture
    def max_delivery_count(self):
        # infra/local/servicebus/config.json: index-jobs MaxDeliveryCount.
        return 5

    @pytest.fixture
    def quiet_period(self):
        return 3.0
