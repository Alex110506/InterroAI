"""
What every `Accounts` must do identically: a dict in the unit tests, Postgres in the cloud.

Subclass `AccountsContract` and provide `accounts`. Times are passed in rather
than read from a clock, so expiry is tested exactly.
"""
from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from cloud.db.accounts import RedeemedCode, RefreshOutcome

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=5)
_github_ids = itertools.count(10_000)


async def _user(accounts, login: str = "octocat"):
    return await accounts.upsert_github_user(next(_github_ids), login, None)


class AccountsContract:
    # ── Users ────────────────────────────────────────────────────────────────

    async def test_signing_in_again_updates_the_same_user(self, accounts):
        first = await accounts.upsert_github_user(42, "old-name", None)
        second = await accounts.upsert_github_user(42, "new-name", "https://avatars.example/42")

        assert second.id == first.id
        assert (second.login, second.avatar_url) == ("new-name", "https://avatars.example/42")
        assert await accounts.get_user(first.id) == second

    async def test_different_github_accounts_are_different_users(self, accounts):
        assert (await _user(accounts)).id != (await _user(accounts)).id

    async def test_an_unknown_user_is_none(self, accounts):
        assert await accounts.get_user(str(uuid4())) is None

    # ── Login codes ──────────────────────────────────────────────────────────

    async def test_a_login_code_redeems_exactly_once(self, accounts):
        user = await _user(accounts)
        await accounts.add_login_code(
            "code-1", user_id=user.id, code_challenge="the-challenge", expires_at=LATER
        )

        assert await accounts.redeem_login_code("code-1", now=NOW) == RedeemedCode(
            user_id=user.id, code_challenge="the-challenge"
        )
        assert await accounts.redeem_login_code("code-1", now=NOW) is None

    async def test_an_expired_login_code_does_not_redeem(self, accounts):
        user = await _user(accounts)
        await accounts.add_login_code("code-1", user_id=user.id, code_challenge="c", expires_at=NOW)
        assert await accounts.redeem_login_code("code-1", now=NOW) is None

    async def test_an_unknown_login_code_does_not_redeem(self, accounts):
        assert await accounts.redeem_login_code("never-issued", now=NOW) is None

    # ── Refresh tokens ───────────────────────────────────────────────────────

    async def test_a_refresh_token_is_spent_by_its_first_use(self, accounts):
        user = await _user(accounts)
        await accounts.add_refresh_token("token-1", user_id=user.id, expires_at=LATER)

        assert await accounts.use_refresh_token("token-1", now=NOW) == RefreshOutcome("ok", user.id)
        assert await accounts.use_refresh_token("token-1", now=NOW) == RefreshOutcome(
            "reused", user.id
        )

    async def test_an_expired_refresh_token_is_invalid(self, accounts):
        user = await _user(accounts)
        await accounts.add_refresh_token("token-1", user_id=user.id, expires_at=NOW)
        assert (await accounts.use_refresh_token("token-1", now=NOW)).status == "invalid"

    async def test_an_unknown_refresh_token_is_invalid(self, accounts):
        assert await accounts.use_refresh_token("never-issued", now=NOW) == RefreshOutcome(
            "invalid"
        )

    async def test_a_revoked_refresh_token_counts_as_reused(self, accounts):
        user = await _user(accounts)
        await accounts.add_refresh_token("token-1", user_id=user.id, expires_at=LATER)

        await accounts.revoke_refresh_token("token-1", now=NOW)
        await accounts.revoke_refresh_token("token-1", now=NOW)  # twice is harmless

        assert (await accounts.use_refresh_token("token-1", now=NOW)).status == "reused"

    # ── Housekeeping ─────────────────────────────────────────────────────────

    async def test_purging_drops_what_has_lapsed_and_keeps_what_has_not(self, accounts):
        user = await _user(accounts)
        await accounts.add_login_code(
            "live", user_id=user.id, code_challenge="the-challenge", expires_at=LATER
        )
        await accounts.add_refresh_token("live-token", user_id=user.id, expires_at=LATER)
        await accounts.add_login_code("lapsed", user_id=user.id, code_challenge="c", expires_at=NOW)
        await accounts.add_refresh_token("lapsed-token", user_id=user.id, expires_at=NOW)

        assert await accounts.purge_expired(now=LATER) == 2

        assert await accounts.redeem_login_code("live", now=NOW) == RedeemedCode(
            user_id=user.id, code_challenge="the-challenge"
        )
        assert await accounts.use_refresh_token("live-token", now=NOW) == RefreshOutcome(
            "ok", user.id
        )

    async def test_revoking_all_ends_every_token_of_that_user_only(self, accounts):
        alice, bob = await _user(accounts, "alice"), await _user(accounts, "bob")
        for token, owner in (("a1", alice), ("a2", alice), ("b1", bob)):
            await accounts.add_refresh_token(token, user_id=owner.id, expires_at=LATER)

        await accounts.revoke_all_refresh_tokens(alice.id, now=NOW)

        assert (await accounts.use_refresh_token("a1", now=NOW)).status == "reused"
        assert (await accounts.use_refresh_token("a2", now=NOW)).status == "reused"
        assert (await accounts.use_refresh_token("b1", now=NOW)).status == "ok"
