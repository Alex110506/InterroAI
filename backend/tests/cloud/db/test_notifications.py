"""
The LISTEN connection that wakes job event streams.

Without a database it degrades to nothing; against real Postgres
(`pytest -m integration`) a NOTIFY reaches the subscribers of its job only.
"""
from __future__ import annotations

import asyncio

import pytest

from cloud.db.jobs import JobRepository
from cloud.db.notifications import JobNotifications, asyncpg_dsn
from cloud.db.session import service_scope
from contracts.indexing import IndexEvent


async def test_without_a_database_a_subscription_only_times_out():
    notifications = JobNotifications(None)
    async with notifications.subscribe("job-1") as signal:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(signal.wait(), 0.01)


def test_the_database_url_becomes_an_asyncpg_dsn():
    dsn = asyncpg_dsn("postgresql+asyncpg://app:s3cret@db.example:5432/interroai")
    assert dsn == "postgresql://app:s3cret@db.example:5432/interroai"


@pytest.mark.integration
async def test_a_notify_wakes_the_subscribers_of_that_job_only(
    app_sessions, database_urls, pg_new_project
):
    jobs = JobRepository(lambda: service_scope(app_sessions))
    mine = await jobs.create(await pg_new_project(), "uploads/mine")
    theirs = await jobs.create(await pg_new_project(), "uploads/theirs")
    notifications = JobNotifications(asyncpg_dsn(database_urls.app))
    try:
        async with (
            notifications.subscribe(mine) as my_signal,
            notifications.subscribe(theirs) as their_signal,
        ):
            await jobs.append_event(mine, IndexEvent(step="C", status="start", total=1))

            await asyncio.wait_for(my_signal.wait(), 5)
            assert not their_signal.is_set()
    finally:
        await notifications.close()
