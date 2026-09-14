"""
Redis cache for derived data — an accelerator, never a dependency.

Two things live here, and both are *recomputable*:

  * **The AST repo map**, keyed by a fingerprint of the files it was built
    from. `build_repo_map` parses every source file in the project, and the
    Coder Agent needs it on every single request; between two requests the
    repo has almost always not changed at all.
  * **Chunk embeddings**, keyed by a hash of the chunk text itself. Content
    addressing is what makes a renamed or moved file free to re-index: the
    bytes are identical, so the vector is already here.

Losing either costs time, never correctness. That is precisely what makes them
safe to keep in evictable storage, and it is why the file→hash manifest that
drives incremental indexing deliberately does **not** live here — that has to
stay consistent with the vectors it describes, so it lives in the vector
store's own metadata (see `core/index/vector_store.py::stored_manifest`).

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
from array import array

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
            # Vectors are packed binary — decoding responses as UTF-8 would
            # corrupt them. Text values are decoded explicitly instead.
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


# ── Embedding vectors ────────────────────────────────────────────────────────


def vector_key(text: str, model: str) -> str:
    """
    Content-addressed key for one chunk's embedding.

    The model is part of the key: two models produce incompatible vectors for
    identical text, and silently mixing them would corrupt every search.
    """
    return f"{_PREFIX}:vec:{model}:{_digest(text)}"


def _encode(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _decode(raw: bytes) -> list[float]:
    values = array("f")
    values.frombytes(raw)
    return values.tolist()


async def get_vectors(keys: list[str]) -> dict[str, list[float]]:
    """Fetch what is cached for *keys*. Missing or unreadable keys are absent."""
    client = _redis()
    if client is None or not keys:
        return {}
    try:
        raw_values = await client.mget(keys)
    except (RedisError, OSError) as exc:
        _give_up(exc)
        return {}

    found: dict[str, list[float]] = {}
    # MGET answers one value per key, in order.
    for key, raw in zip(keys, raw_values, strict=True):
        if raw is None:
            continue
        try:
            found[key] = _decode(raw)
        except (ValueError, TypeError):
            # A truncated or foreign value is a miss, not a crash: the caller
            # recomputes and overwrites it.
            logger.debug("Discarding an undecodable cached vector for %s.", key)
    return found


async def set_vectors(vectors: dict[str, list[float]]) -> None:
    """Store *vectors* by key, each with its own TTL."""
    client = _redis()
    if client is None or not vectors:
        return
    try:
        # A pipeline so one round trip covers the whole batch; MSET cannot
        # carry a per-key TTL.
        pipe = client.pipeline(transaction=False)
        for key, vector in vectors.items():
            pipe.set(key, _encode(vector), ex=_TTL_SECONDS)
        await pipe.execute()
    except (RedisError, OSError) as exc:
        _give_up(exc)
