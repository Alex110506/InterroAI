"""
Rate limits: how many requests one caller may make per minute.

Quotas (`cloud/db/usage.py`) cap what a user spends in a day. Rate limits stop a
burst, such as a runaway client or a script hammering sign-in, from spending it
in a minute. Each limit is a token bucket per caller: it refills at `per_minute`
tokens a minute, and a caller may use that many at once.

The buckets live in this process's memory. Each API replica counts on its own
and a restart forgets them, so with two replicas a caller gets up to twice the
limit (ADR 0008). That is good enough for what the limits are for; counting
exactly across replicas would need a shared store such as Redis, which this
project's budget leaves out.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

#: A bucket untouched this long has refilled completely.
_FULL_AFTER_SECONDS = 60.0


@dataclass(frozen=True)
class RateLimits:
    """Requests per minute, per caller."""

    sign_in: int = 30
    chat: int = 60
    search: int = 120
    sync: int = 30
    uploads: int = 30


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic, max_buckets: int = 50_000
    ) -> None:
        self._clock = clock
        self._max_buckets = max_buckets
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def __len__(self) -> int:
        return len(self._buckets)

    def hit(self, limit: str, caller: str, per_minute: int) -> float:
        """
        Spend one request from *caller*'s bucket for *limit*.

        Returns 0 when the request may go ahead, and otherwise the seconds until
        it could.
        """
        if per_minute <= 0:
            return _FULL_AFTER_SECONDS
        now = self._clock()
        refill_per_second = per_minute / 60
        key = (limit, caller)

        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_buckets:
                self._evict(now)
            bucket = self._buckets[key] = _Bucket(tokens=per_minute, updated=now)
        else:
            elapsed = now - bucket.updated
            bucket.tokens = min(per_minute, bucket.tokens + elapsed * refill_per_second)
            bucket.updated = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return 0.0
        return (1 - bucket.tokens) / refill_per_second

    def _evict(self, now: float) -> None:
        # A full bucket behaves exactly like no bucket, so dropping the idle
        # ones loses nothing.
        for key in [k for k, b in self._buckets.items() if now - b.updated >= _FULL_AFTER_SECONDS]:
            del self._buckets[key]
        if len(self._buckets) >= self._max_buckets:
            # Still full: drop the least recently used half. They start afresh.
            by_age = sorted(self._buckets, key=lambda k: self._buckets[k].updated)
            for key in by_age[: len(by_age) // 2]:
                del self._buckets[key]
