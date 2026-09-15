"""
Users, and the credentials that sign them in: login codes and refresh tokens.

These tables sit outside row-level security, since sign-in happens before anyone
is known, so every query here carries all of its own conditions. Each
credential is spent by a single conditional UPDATE: two requests racing with the
same code or token cannot both succeed.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from cloud.db.models import LoginCode, RefreshToken, User
from cloud.db.session import ScopeFactory


@dataclass(frozen=True)
class UserRecord:
    id: str
    github_id: int
    login: str
    avatar_url: str | None


@dataclass(frozen=True)
class RedeemedCode:
    user_id: str
    code_challenge: str


@dataclass(frozen=True)
class RefreshOutcome:
    """
    What presenting a refresh token found.

      * `ok`      — it was live, and is now spent.
      * `reused`  — it had already been spent or revoked. Whoever presents it
                    is not necessarily who it was issued to.
      * `invalid` — unknown, or expired.
    """

    status: Literal["ok", "reused", "invalid"]
    user_id: str | None = None


class Accounts(Protocol):
    async def upsert_github_user(
        self, github_id: int, login: str, avatar_url: str | None
    ) -> UserRecord: ...

    async def get_user(self, user_id: str) -> UserRecord | None: ...

    async def add_login_code(
        self, code_hash: str, *, user_id: str, code_challenge: str, expires_at: datetime
    ) -> None: ...

    async def redeem_login_code(self, code_hash: str, *, now: datetime) -> RedeemedCode | None:
        """Spend a live code. `None` if it is unknown, used or expired."""
        ...

    async def add_refresh_token(
        self, token_hash: str, *, user_id: str, expires_at: datetime
    ) -> None: ...

    async def use_refresh_token(self, token_hash: str, *, now: datetime) -> RefreshOutcome: ...

    async def revoke_refresh_token(self, token_hash: str, *, now: datetime) -> None: ...

    async def revoke_all_refresh_tokens(self, user_id: str, *, now: datetime) -> None: ...


class PostgresAccounts:
    def __init__(self, scope: ScopeFactory) -> None:
        self._scope = scope

    async def upsert_github_user(
        self, github_id: int, login: str, avatar_url: str | None
    ) -> UserRecord:
        statement = (
            insert(User)
            .values(id=uuid.uuid4(), github_id=github_id, login=login, avatar_url=avatar_url)
            .on_conflict_do_update(
                index_elements=["github_id"],
                # A renamed GitHub account keeps its id, so the row follows the rename.
                set_={"login": login, "avatar_url": avatar_url},
            )
            .returning(User.id, User.github_id, User.login, User.avatar_url)
        )
        async with self._scope() as session:
            row = (await session.execute(statement)).one()
        return _user(row)

    async def get_user(self, user_id: str) -> UserRecord | None:
        try:
            key = uuid.UUID(user_id)
        except ValueError:
            return None
        statement = select(User.id, User.github_id, User.login, User.avatar_url).where(
            User.id == key
        )
        async with self._scope() as session:
            row = (await session.execute(statement)).one_or_none()
        return None if row is None else _user(row)

    async def add_login_code(
        self, code_hash: str, *, user_id: str, code_challenge: str, expires_at: datetime
    ) -> None:
        async with self._scope() as session:
            session.add(
                LoginCode(
                    code_hash=code_hash,
                    user_id=uuid.UUID(user_id),
                    code_challenge=code_challenge,
                    expires_at=expires_at,
                )
            )

    async def redeem_login_code(self, code_hash: str, *, now: datetime) -> RedeemedCode | None:
        statement = (
            update(LoginCode)
            .where(
                LoginCode.code_hash == code_hash,
                LoginCode.used_at.is_(None),
                LoginCode.expires_at > now,
            )
            .values(used_at=now)
            .returning(LoginCode.user_id, LoginCode.code_challenge)
        )
        async with self._scope() as session:
            row = (await session.execute(statement)).one_or_none()
        if row is None:
            return None
        return RedeemedCode(user_id=str(row.user_id), code_challenge=row.code_challenge)

    async def add_refresh_token(
        self, token_hash: str, *, user_id: str, expires_at: datetime
    ) -> None:
        async with self._scope() as session:
            session.add(
                RefreshToken(
                    id=uuid.uuid4(),
                    user_id=uuid.UUID(user_id),
                    token_hash=token_hash,
                    expires_at=expires_at,
                )
            )

    async def use_refresh_token(self, token_hash: str, *, now: datetime) -> RefreshOutcome:
        spend = (
            update(RefreshToken)
            .where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at.is_(None),
                RefreshToken.expires_at > now,
            )
            .values(revoked_at=now)
            .returning(RefreshToken.user_id)
        )
        async with self._scope() as session:
            spent_by = (await session.execute(spend)).scalar_one_or_none()
            if spent_by is not None:
                return RefreshOutcome("ok", str(spent_by))
            # Not spendable. Tell apart "someone already used it" from "no such
            # live token": only the first one means the token has leaked.
            row = (
                await session.execute(
                    select(RefreshToken.user_id, RefreshToken.revoked_at).where(
                        RefreshToken.token_hash == token_hash
                    )
                )
            ).one_or_none()
        if row is None:
            return RefreshOutcome("invalid")
        if row.revoked_at is not None:
            return RefreshOutcome("reused", str(row.user_id))
        return RefreshOutcome("invalid", str(row.user_id))

    async def revoke_refresh_token(self, token_hash: str, *, now: datetime) -> None:
        async with self._scope() as session:
            await session.execute(
                update(RefreshToken)
                .where(RefreshToken.token_hash == token_hash, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=now)
            )

    async def revoke_all_refresh_tokens(self, user_id: str, *, now: datetime) -> None:
        async with self._scope() as session:
            await session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.user_id == uuid.UUID(user_id),
                    RefreshToken.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )


def _user(row) -> UserRecord:
    return UserRecord(
        id=str(row.id), github_id=row.github_id, login=row.login, avatar_url=row.avatar_url
    )
