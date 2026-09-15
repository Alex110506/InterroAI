"""The Postgres `UsageMeter`. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import asyncio

import pytest
from port_contracts.usage import DAY, UsageMeterContract

from cloud.db.session import anonymous_scope
from cloud.db.usage import PostgresUsageMeter, Quota

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_meter(app_sessions):
    return PostgresUsageMeter(lambda: anonymous_scope(app_sessions))


class TestPostgresUsageMeter(UsageMeterContract):
    @pytest.fixture
    def meter(self, pg_meter):
        return pg_meter

    @pytest.fixture
    async def user_id(self, pg_new_user):
        return await pg_new_user("metered")

    @pytest.fixture
    async def other_user_id(self, pg_new_user):
        return await pg_new_user("also-metered")


async def test_racing_requests_cannot_overshoot_the_quota(pg_meter, pg_new_user):
    user_id = await pg_new_user("racer")

    admitted = await asyncio.gather(
        *(pg_meter.admit(user_id, DAY, Quota(requests=3, tokens=10**6)) for _ in range(8))
    )

    assert sum(admitted) == 3
    assert (await pg_meter.totals(user_id, DAY)).requests == 3
