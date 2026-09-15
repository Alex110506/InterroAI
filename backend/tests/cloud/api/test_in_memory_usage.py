"""The in-memory `UsageMeter` the gateway tests run on, held to the shared contract."""
from __future__ import annotations

from uuid import uuid4

import pytest
from fakes.usage import InMemoryUsageMeter
from port_contracts.usage import UsageMeterContract


class TestInMemoryUsageMeter(UsageMeterContract):
    @pytest.fixture
    def meter(self):
        return InMemoryUsageMeter()

    @pytest.fixture
    def user_id(self):
        return str(uuid4())

    @pytest.fixture
    def other_user_id(self):
        return str(uuid4())
