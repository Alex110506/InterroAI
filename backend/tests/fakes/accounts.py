"""
An in-memory `Accounts`, so identity can be tested without Postgres.

It is held to the same contract suite as `PostgresAccounts`
(`tests/port_contracts/accounts.py`), so the two cannot quietly drift apart.
"""
from __future__ import annotations

from uuid import uuid4

from cloud.auth.accounts import UserRecord


class InMemoryAccounts:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}

    async def upsert_github_user(
        self, github_id: int, login: str, avatar_url: str | None
    ) -> UserRecord:
        existing = await self.get_by_github_id(github_id)
        user = UserRecord(
            id=existing.id if existing else str(uuid4()),
            github_id=github_id,
            login=login,
            avatar_url=avatar_url,
        )
        self.users[user.id] = user
        return user

    async def get_user(self, user_id: str) -> UserRecord | None:
        return self.users.get(user_id)

    async def get_by_github_id(self, github_id: int) -> UserRecord | None:
        return next((u for u in self.users.values() if u.github_id == github_id), None)
