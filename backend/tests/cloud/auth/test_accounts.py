"""The Postgres `Accounts`. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import asyncio

import pytest
from port_contracts.accounts import AccountsContract
from sqlalchemy import text

from cloud.auth.accounts import PostgresAccounts
from cloud.db.session import anonymous_scope

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_accounts(app_sessions):
    return PostgresAccounts(lambda: anonymous_scope(app_sessions))


class TestPostgresAccounts(AccountsContract):
    @pytest.fixture
    def accounts(self, pg_accounts):
        return pg_accounts


async def test_two_first_requests_racing_create_one_user(pg_accounts):
    """Two requests from a brand-new account can arrive together; one row comes out."""
    users = await asyncio.gather(
        *(pg_accounts.upsert_github_user(7, "racer", None) for _ in range(5))
    )
    assert len({user.id for user in users}) == 1


async def test_the_sign_in_scope_sees_no_ones_projects(app_sessions, pg_new_project):
    await pg_new_project()
    async with anonymous_scope(app_sessions) as session:
        assert (await session.execute(text("SELECT count(*) FROM projects"))).scalar_one() == 0
