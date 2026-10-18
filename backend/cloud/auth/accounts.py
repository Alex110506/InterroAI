"""
Users: one row per GitHub account that has used InterroAI.

Sign-in itself lives at Auth0, so no credential is stored here: a row is the
GitHub id the user is known by, the login and avatar last read from Auth0, and
the internal id every project hangs off. The table sits outside row-level
security, since a user has to be found before anyone is known.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from cloud.db.models import User
from cloud.db.session import ScopeFactory


@dataclass(frozen=True)
class UserRecord:
    id: str
    github_id: int
    login: str
    avatar_url: str | None


class Accounts(Protocol):
    async def upsert_github_user(
        self, github_id: int, login: str, avatar_url: str | None
    ) -> UserRecord: ...

    async def get_user(self, user_id: str) -> UserRecord | None: ...

    async def get_by_github_id(self, github_id: int) -> UserRecord | None: ...


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
        return await self._one(User.id == key)

    async def get_by_github_id(self, github_id: int) -> UserRecord | None:
        return await self._one(User.github_id == github_id)

    async def _one(self, condition) -> UserRecord | None:
        statement = select(User.id, User.github_id, User.login, User.avatar_url).where(condition)
        async with self._scope() as session:
            row = (await session.execute(statement)).one_or_none()
        return None if row is None else _user(row)


def _user(row) -> UserRecord:
    return UserRecord(
        id=str(row.id), github_id=row.github_id, login=row.login, avatar_url=row.avatar_url
    )
