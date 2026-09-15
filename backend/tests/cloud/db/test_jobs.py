"""Index job rows. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import asyncio

import asyncpg
import pytest
from sqlalchemy.engine import make_url

from cloud.db.jobs import NOTIFY_CHANNEL, ActiveJobExistsError, JobRepository
from cloud.db.session import service_scope, user_scope
from contracts.indexing import IndexEvent

pytestmark = pytest.mark.integration


@pytest.fixture
def jobs(app_sessions):
    return JobRepository(lambda: service_scope(app_sessions))


async def test_a_new_job_is_queued_with_no_events(jobs, pg_new_project):
    job = await jobs.get(await jobs.create(await pg_new_project(), "uploads/one"))
    assert (job.status, job.events, job.error) == ("queued", [], None)


async def test_events_are_appended_in_order_and_counted(jobs, pg_new_project):
    job_id = await jobs.create(await pg_new_project(), "uploads/one")

    assert await jobs.append_event(job_id, IndexEvent(step="C", status="start", total=3)) == 1
    assert await jobs.append_event(job_id, IndexEvent(step="D", status="done", deleted=0)) == 2

    job = await jobs.get(job_id)
    assert job.events == [
        {"step": "C", "status": "start", "total": 3},
        {"step": "D", "status": "done", "deleted": 0},
    ]


async def test_finishing_records_done_or_failed(jobs, pg_new_project):
    done_id = await jobs.create(await pg_new_project(), "uploads/done")
    failed_id = await jobs.create(await pg_new_project(), "uploads/failed")

    await jobs.finish(done_id)
    await jobs.finish(failed_id, error="provider refused")

    assert (await jobs.get(done_id)).status == "done"
    failed = await jobs.get(failed_id)
    assert (failed.status, failed.error) == ("failed", "provider refused")


async def test_a_project_has_at_most_one_active_job(jobs, pg_new_project):
    project = await pg_new_project()
    first = await jobs.create(project, "uploads/first")

    with pytest.raises(ActiveJobExistsError) as raised:
        await jobs.create(project, "uploads/second")

    assert raised.value.job_id == first


async def test_a_finished_job_frees_its_project_for_the_next(jobs, pg_new_project):
    project = await pg_new_project()
    await jobs.finish(await jobs.create(project, "uploads/first"))
    assert await jobs.create(project, "uploads/second")


async def test_appending_an_event_notifies_listeners(jobs, pg_new_project, database_urls):
    job_id = await jobs.create(await pg_new_project(), "uploads/one")
    dsn = make_url(database_urls.owner).set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    heard: asyncio.Queue[str] = asyncio.Queue()
    connection = await asyncpg.connect(dsn)
    try:
        await connection.add_listener(
            NOTIFY_CHANNEL, lambda *args: heard.put_nowait(args[-1])
        )
        await jobs.append_event(job_id, IndexEvent(step="C", status="start", total=1))
        assert await asyncio.wait_for(heard.get(), 5) == job_id
    finally:
        await connection.close()


async def test_a_user_cannot_see_another_users_job(
    app_sessions, jobs, pg_new_user, pg_new_project
):
    job_id = await jobs.create(await pg_new_project(await pg_new_user("owner")), "uploads/x")
    stranger = await pg_new_user("stranger")

    as_stranger = JobRepository(lambda: user_scope(app_sessions, stranger))
    assert await as_stranger.get(job_id) is None
