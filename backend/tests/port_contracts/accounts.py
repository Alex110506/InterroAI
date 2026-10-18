"""
What every `Accounts` must do identically: a dict in the unit tests, Postgres in the cloud.

Subclass `AccountsContract` and provide `accounts`.
"""
from __future__ import annotations

import itertools
from uuid import uuid4

_github_ids = itertools.count(10_000)


async def _user(accounts, login: str = "octocat"):
    return await accounts.upsert_github_user(next(_github_ids), login, None)


class AccountsContract:
    async def test_signing_in_again_updates_the_same_user(self, accounts):
        first = await accounts.upsert_github_user(42, "old-name", None)
        second = await accounts.upsert_github_user(42, "new-name", "https://avatars.example/42")

        assert second.id == first.id
        assert (second.login, second.avatar_url) == ("new-name", "https://avatars.example/42")
        assert await accounts.get_user(first.id) == second

    async def test_different_github_accounts_are_different_users(self, accounts):
        assert (await _user(accounts)).id != (await _user(accounts)).id

    async def test_a_user_is_found_by_github_id(self, accounts):
        user = await _user(accounts, "found-me")
        assert await accounts.get_by_github_id(user.github_id) == user

    async def test_an_unknown_github_id_is_none(self, accounts):
        assert await accounts.get_by_github_id(next(_github_ids)) is None

    async def test_an_unknown_user_is_none(self, accounts):
        assert await accounts.get_user(str(uuid4())) is None

    async def test_a_user_id_that_is_not_a_uuid_is_none(self, accounts):
        assert await accounts.get_user("not-a-uuid") is None
