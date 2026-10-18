"""
Per-user, per-day use of the platform's OpenAI key, and the quotas on it.

A request is admitted and counted in one statement, so two requests racing for a
user's last allowance cannot both get it. Tokens are known only once a response
has finished, so they are added afterwards: a user can overshoot the token quota
by the one response in flight, never by more.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from sqlalchemy import text

from cloud.db.session import ScopeFactory


@dataclass(frozen=True)
class Quota:
    requests: int
    tokens: int


@dataclass(frozen=True)
class UsageTotals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class UsageMeter(Protocol):
    async def admit(self, user_id: str, day: date, quota: Quota) -> bool:
        """Count one request if the user is under both quotas for *day*; False if not."""
        ...

    async def add_tokens(
        self, user_id: str, day: date, *, prompt: int, completion: int
    ) -> None: ...

    async def totals(self, user_id: str, day: date) -> UsageTotals: ...


class PostgresUsageMeter:
    def __init__(self, scope: ScopeFactory) -> None:
        self._scope = scope

    async def admit(self, user_id: str, day: date, quota: Quota) -> bool:
        if quota.requests <= 0 or quota.tokens <= 0:
            return False
        # ON CONFLICT DO UPDATE locks the row and re-checks the WHERE against
        # its latest version, so concurrent requests queue up rather than all
        # reading the same count.
        statement = text(
            "INSERT INTO usage (user_id, day, requests, prompt_tokens, completion_tokens) "
            "VALUES (:user_id, :day, 1, 0, 0) "
            "ON CONFLICT (user_id, day) DO UPDATE SET requests = usage.requests + 1 "
            "WHERE usage.requests < :max_requests "
            "  AND usage.prompt_tokens + usage.completion_tokens < :max_tokens "
            "RETURNING requests"
        )
        async with self._scope() as session:
            admitted = await session.scalar(
                statement,
                {
                    "user_id": uuid.UUID(user_id),
                    "day": day,
                    "max_requests": quota.requests,
                    "max_tokens": quota.tokens,
                },
            )
        return admitted is not None

    async def add_tokens(self, user_id: str, day: date, *, prompt: int, completion: int) -> None:
        statement = text(
            "INSERT INTO usage (user_id, day, requests, prompt_tokens, completion_tokens) "
            "VALUES (:user_id, :day, 0, :prompt, :completion) "
            "ON CONFLICT (user_id, day) DO UPDATE "
            "SET prompt_tokens = usage.prompt_tokens + EXCLUDED.prompt_tokens, "
            "    completion_tokens = usage.completion_tokens + EXCLUDED.completion_tokens"
        )
        async with self._scope() as session:
            await session.execute(
                statement,
                {
                    "user_id": uuid.UUID(user_id),
                    "day": day,
                    "prompt": prompt,
                    "completion": completion,
                },
            )

    async def totals(self, user_id: str, day: date) -> UsageTotals:
        statement = text(
            "SELECT requests, prompt_tokens, completion_tokens FROM usage "
            "WHERE user_id = :user_id AND day = :day"
        )
        async with self._scope() as session:
            row = (
                await session.execute(statement, {"user_id": uuid.UUID(user_id), "day": day})
            ).one_or_none()
        if row is None:
            return UsageTotals()
        return UsageTotals(
            requests=row.requests,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
        )
