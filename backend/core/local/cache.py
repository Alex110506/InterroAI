"""
Redis cache for derived data — an accelerator, never a dependency.

One thing lives here, and it is *recomputable*: the **AST repo map**, keyed by
a fingerprint of the files it was built from. `build_repo_map` parses every
source file in the project, and the Coder Agent needs it on every single
request; between two requests the repo has almost always not changed at all.

Losing it costs time, never correctness. That is precisely what makes it safe
to keep in evictable storage, and it is why the file→hash manifest that drives
incremental indexing deliberately does **not** live here — that has to stay
consistent with the vectors it describes, so it lives with them, in the index's
own store. Chunk embeddings are the cloud's business for the same reason the
vectors are: the worker caches those in Postgres, next to the platform key, so
nothing on this machine holds a vector.

If no Redis is reachable, every read misses and every write is dropped, so the
caller simply recomputes. A desktop app must not stop working because a
background service is down, so the first failure is logged once and the cache
then reports itself unavailable rather than retrying (and paying the connect
timeout) on every subsequent call.
"""
from __future__ import annotations

import hashlib
import logging
import os

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

#: Overridable so a non-default port or a second instance can be pointed at.
_URL_ENV = "INTERROAI_REDIS_URL"
_DEFAULT_URL = "redis://localhost:6379/0"

#: Key prefix carries a schema version: if the encoding below ever changes,
#: bumping this abandons the old keys instead of misreading them.
_PREFIX = "interroai:v1"

#: Entries are self-invalidating (both key spaces are content- or
#: fingerprint-addressed), so the TTL is only there to stop dead keys from
#: accumulating forever after a file changes.
_TTL_SECONDS = 30 * 24 * 60 * 60

#: Fail fast. A cache that blocks the app for seconds is worse than no cache.
_CONNECT_TIMEOUT = 0.5

_client: Redis | None = None
_unavailable = False


def _redis() -> Redis | None:
    """The shared client, or None once Redis has been found unreachable."""
    global _client
    if _unavailable:
        return None
    if _client is None:
        # `from_url` does not connect; the first command does. Nothing here
        # can fail yet, so unavailability is discovered at the first use.
        _client = Redis.from_url(
            os.environ.get(_URL_ENV, _DEFAULT_URL),
            # Decoded explicitly at each call site instead, so a future
            # binary value cannot be silently mangled by a UTF-8 round trip.
            decode_responses=False,
            socket_connect_timeout=_CONNECT_TIMEOUT,
            socket_timeout=_CONNECT_TIMEOUT,
        )
    return _client


def _give_up(exc: BaseException) -> None:
    """Mark the cache unavailable, explaining itself exactly once."""
    global _unavailable
    if not _unavailable:
        _unavailable = True
        logger.warning(
            "Redis is not available (%s) — continuing without the cache. "
            "Start a local redis-server, or set %s, to speed up re-indexing.",
            exc,
            _URL_ENV,
        )


async def reset() -> None:
    """Forget the client and the unavailable flag. For tests and key rotation."""
    global _client, _unavailable
    client, _client, _unavailable = _client, None, False
    if client is not None:
        try:
            await client.aclose()
        except (RedisError, OSError):
            pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


# ── Repo map ─────────────────────────────────────────────────────────────────


def _repo_map_key(project_path: str, fingerprint: str) -> str:
    # The fingerprint is part of the key rather than a value to compare, so a
    # changed project is a plain cache miss — there is no stale entry to
    # detect, and the old key simply ages out.
    return f"{_PREFIX}:repomap:{_digest(project_path)}:{fingerprint}"


async def get_repo_map(project_path: str, fingerprint: str) -> str | None:
    """The cached repo map for this exact project state, or None."""
    client = _redis()
    if client is None:
        return None
    try:
        raw = await client.get(_repo_map_key(project_path, fingerprint))
    except (RedisError, OSError) as exc:
        _give_up(exc)
        return None
    return None if raw is None else raw.decode("utf-8", errors="replace")


async def set_repo_map(project_path: str, fingerprint: str, repo_map: str) -> None:
    client = _redis()
    if client is None:
        return
    try:
        await client.set(
            _repo_map_key(project_path, fingerprint),
            repo_map.encode("utf-8"),
            ex=_TTL_SECONDS,
        )
    except (RedisError, OSError) as exc:
        _give_up(exc)


