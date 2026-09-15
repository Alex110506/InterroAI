"""
Engines, and the scopes every unit of database work runs inside.

A scope opens a transaction and tells Postgres who is asking. The row-level
security policies from migration 0001 read that:

  * `user_scope(sessions, user_id)` — the API, on a user's behalf. Queries see
    only that user's projects, chunks and jobs *however they are written*: a
    forgotten `WHERE owner_id = …` returns nothing, not someone else's code.
  * `service_scope(sessions)` — the worker, which acts for whoever owns the job
    it is running and always addresses rows by the project on that job.
  * `anonymous_scope(sessions)` — sign-in, which happens before anyone is known.
    It sees no project, chunk or job at all.

Both identities are transaction-local (`set_config(…, true)`), so they vanish
at commit and a pooled connection can never carry one request's identity into
the next.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

#: What the adapters take: a way to open a scope, so one class serves both the
#: API (user scopes) and the worker (service scopes).
ScopeFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def create_engine(database_url: str) -> AsyncEngine:
    # pre_ping: Azure closes idle connections; a dead pooled one should be
    # replaced, not surface as a failed request.
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def user_scope(
    sessions: async_sessionmaker[AsyncSession], user_id: uuid.UUID | str
) -> AsyncIterator[AsyncSession]:
    """A transaction in which row-level security admits only *user_id*'s rows."""
    async with sessions() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.user_id', :user_id, true)"), {"user_id": str(user_id)}
        )
        yield session


@asynccontextmanager
async def service_scope(sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """A transaction for the worker, which acts across tenants by project id."""
    async with sessions() as session, session.begin():
        await session.execute(text("SELECT set_config('app.service', 'worker', true)"))
        yield session


@asynccontextmanager
async def anonymous_scope(
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """
    A transaction on nobody's behalf, for the sign-in tables.

    Those tables sit outside row-level security, so sign-in needs no identity —
    and is given none, rather than borrowing the worker's: a bug in sign-in
    code can then read no one's projects.
    """
    async with sessions() as session, session.begin():
        yield session


def as_float_list(value) -> list[float]:
    """A vector column's value as a plain list, whatever the driver handed back."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "to_list"):
        value = value.to_list()
    return [float(component) for component in value]
