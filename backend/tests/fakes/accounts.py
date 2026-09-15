"""
An in-memory `Accounts`, so the sign-in flow can be tested without Postgres.

It is held to the same contract suite as `PostgresAccounts`
(`tests/port_contracts/accounts.py`), so the two cannot quietly drift apart.
"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from cloud.db.accounts import RedeemedCode, RefreshOutcome, UserRecord


class InMemoryAccounts:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}
        self.login_codes: dict[str, dict] = {}
        self.refresh_tokens: dict[str, dict] = {}

    async def upsert_github_user(
        self, github_id: int, login: str, avatar_url: str | None
    ) -> UserRecord:
        existing = next((u for u in self.users.values() if u.github_id == github_id), None)
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

    async def add_login_code(
        self, code_hash: str, *, user_id: str, code_challenge: str, expires_at: datetime
    ) -> None:
        self.login_codes[code_hash] = {
            "user_id": user_id,
            "code_challenge": code_challenge,
            "expires_at": expires_at,
            "used_at": None,
        }

    async def redeem_login_code(self, code_hash: str, *, now: datetime) -> RedeemedCode | None:
        row = self.login_codes.get(code_hash)
        if row is None or row["used_at"] is not None or row["expires_at"] <= now:
            return None
        row["used_at"] = now
        return RedeemedCode(user_id=row["user_id"], code_challenge=row["code_challenge"])

    async def add_refresh_token(
        self, token_hash: str, *, user_id: str, expires_at: datetime
    ) -> None:
        self.refresh_tokens[token_hash] = {
            "user_id": user_id,
            "expires_at": expires_at,
            "revoked_at": None,
        }

    async def use_refresh_token(self, token_hash: str, *, now: datetime) -> RefreshOutcome:
        row = self.refresh_tokens.get(token_hash)
        if row is None:
            return RefreshOutcome("invalid")
        if row["revoked_at"] is not None:
            return RefreshOutcome("reused", row["user_id"])
        if row["expires_at"] <= now:
            return RefreshOutcome("invalid", row["user_id"])
        row["revoked_at"] = now
        return RefreshOutcome("ok", row["user_id"])

    async def revoke_refresh_token(self, token_hash: str, *, now: datetime) -> None:
        row = self.refresh_tokens.get(token_hash)
        if row is not None and row["revoked_at"] is None:
            row["revoked_at"] = now

    async def revoke_all_refresh_tokens(self, user_id: str, *, now: datetime) -> None:
        for row in self.refresh_tokens.values():
            if row["user_id"] == user_id and row["revoked_at"] is None:
                row["revoked_at"] = now
