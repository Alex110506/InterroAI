"""
What every `UsageMeter` must do identically: a dict in the unit tests, Postgres in the cloud.

Subclass `UsageMeterContract` and provide `meter`, and two distinct existing
users as `user_id` and `other_user_id`.
"""
from __future__ import annotations

from datetime import date, timedelta

from cloud.db.usage import Quota, UsageTotals

DAY = date(2026, 9, 15)
QUOTA = Quota(requests=3, tokens=1_000)


class UsageMeterContract:
    async def test_requests_are_admitted_up_to_the_quota(self, meter, user_id):
        admitted = [await meter.admit(user_id, DAY, QUOTA) for _ in range(4)]

        assert admitted == [True, True, True, False]
        assert (await meter.totals(user_id, DAY)).requests == 3

    async def test_reaching_the_token_quota_refuses_further_requests(self, meter, user_id):
        assert await meter.admit(user_id, DAY, QUOTA)
        await meter.add_tokens(user_id, DAY, prompt=600, completion=400)
        assert not await meter.admit(user_id, DAY, QUOTA)

    async def test_tokens_accumulate(self, meter, user_id):
        await meter.add_tokens(user_id, DAY, prompt=10, completion=5)
        await meter.add_tokens(user_id, DAY, prompt=1, completion=2)

        assert await meter.totals(user_id, DAY) == UsageTotals(
            requests=0, prompt_tokens=11, completion_tokens=7
        )

    async def test_each_day_starts_afresh(self, meter, user_id):
        for _ in range(QUOTA.requests):
            await meter.admit(user_id, DAY, QUOTA)
        assert await meter.admit(user_id, DAY + timedelta(days=1), QUOTA)

    async def test_users_do_not_share_a_quota(self, meter, user_id, other_user_id):
        for _ in range(QUOTA.requests):
            await meter.admit(user_id, DAY, QUOTA)
        assert await meter.admit(other_user_id, DAY, QUOTA)

    async def test_a_zero_quota_admits_nothing(self, meter, user_id):
        assert not await meter.admit(user_id, DAY, Quota(requests=0, tokens=1_000))

    async def test_a_day_without_use_totals_zero(self, meter, user_id):
        assert await meter.totals(user_id, DAY) == UsageTotals()
