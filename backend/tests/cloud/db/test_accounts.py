"""The Postgres `Accounts`. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import asyncio

import pytest
from port_contracts.accounts import LATER, NOW, AccountsContract
from sqlalchemy import text

from cloud.db.accounts import PostgresAccounts
from cloud.db.session import anonymous_scope

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_accounts(app_sessions):
    return PostgresAccounts(lambda: anonymous_scope(app_sessions))


class TestPostgresAccounts(AccountsContract):
    @pytest.fixture
    def accounts(self, pg_accounts):
        return pg_accounts


async def test_racing_uses_of_one_refresh_token_let_exactly_one_through(pg_accounts):
    user = await pg_accounts.upsert_github_user(7, "racer", None)
    await pg_accounts.add_refresh_token("contested", user_id=user.id, expires_at=LATER)

    outcomes = await asyncio.gather(
        *(pg_accounts.use_refresh_token("contested", now=NOW) for _ in range(5))
    )

    assert sorted(outcome.status for outcome in outcomes) == ["ok"] + ["reused"] * 4


async def test_racing_redemptions_of_one_login_code_let_exactly_one_through(pg_accounts):
    user = await pg_accounts.upsert_github_user(8, "racer", None)
    await pg_accounts.add_login_code(
        "contested", user_id=user.id, code_challenge="c", expires_at=LATER
    )

    redeemed = await asyncio.gather(
        *(pg_accounts.redeem_login_code("contested", now=NOW) for _ in range(5))
    )

    assert sum(result is not None for result in redeemed) == 1


async def test_purging_really_removes_the_rows(pg_accounts, app_sessions):
    """Both tables only ever grow otherwise: a code per sign-in, a token per refresh."""
    user = await pg_accounts.upsert_github_user(9, "tidy", None)
    await pg_accounts.add_login_code("lapsed", user_id=user.id, code_challenge="c", expires_at=NOW)
    await pg_accounts.add_refresh_token("lapsed", user_id=user.id, expires_at=NOW)

    assert await pg_accounts.purge_expired(now=LATER) == 2

    async with anonymous_scope(app_sessions) as session:
        left = await session.execute(
            text(
                "SELECT (SELECT count(*) FROM login_codes) "
                "+ (SELECT count(*) FROM refresh_tokens)"
            )
        )
        assert left.scalar_one() == 0


async def test_the_sign_in_scope_sees_no_ones_projects(app_sessions, pg_new_project):
    await pg_new_project()
    async with anonymous_scope(app_sessions) as session:
        assert (await session.execute(text("SELECT count(*) FROM projects"))).scalar_one() == 0
