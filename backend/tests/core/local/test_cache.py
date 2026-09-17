"""
The Redis cache.

One thing is cached here: the AST repo map the coder opens with. Embeddings are
the cloud's business — the worker caches those in Postgres, next to the platform
key — so nothing on this machine holds a vector.

Two properties matter more than anything else: that a cached value is the same
value that went in, and that *no* failure of the cache can fail the app. The
in-memory double from `conftest.py` stands in for the server, so these tests
exercise the real keying without one.
"""
from __future__ import annotations

from conftest import FakeRedis
from redis.exceptions import RedisError

import core.local.cache as cache

# ── Repo map ─────────────────────────────────────────────────────────────────


async def test_a_repo_map_survives_the_round_trip():
    await cache.set_repo_map("/proj", "fp-1", "## main.py\ndef f(): ...")
    assert await cache.get_repo_map("/proj", "fp-1") == "## main.py\ndef f(): ..."


async def test_a_new_fingerprint_misses():
    """The fingerprint is the key, so a changed project cannot read a stale map."""
    await cache.set_repo_map("/proj", "fp-1", "old map")
    assert await cache.get_repo_map("/proj", "fp-2") is None


async def test_two_projects_do_not_share_a_map():
    await cache.set_repo_map("/one", "fp", "map one")
    await cache.set_repo_map("/two", "fp", "map two")
    assert await cache.get_repo_map("/one", "fp") == "map one"


async def test_an_empty_map_is_still_a_hit():
    """A project with no parseable sources must not re-walk on every request."""
    await cache.set_repo_map("/proj", "fp", "")
    assert await cache.get_repo_map("/proj", "fp") == ""


async def test_reading_an_unwritten_map_is_a_plain_miss():
    assert await cache.get_repo_map("/never-indexed", "fp") is None


# ── Degradation ──────────────────────────────────────────────────────────────


async def test_reads_miss_when_redis_is_unreachable(fake_redis):
    fake_redis.broken = True
    assert await cache.get_repo_map("/proj", "fp") is None


async def test_writes_are_dropped_when_redis_is_unreachable(fake_redis):
    fake_redis.broken = True
    await cache.set_repo_map("/proj", "fp", "map")  # must not raise


async def test_the_first_failure_explains_itself_once(fake_redis, caplog):
    fake_redis.broken = True
    with caplog.at_level("WARNING", logger="core.local.cache"):
        await cache.get_repo_map("/proj", "fp")
        await cache.get_repo_map("/proj", "fp")
        await cache.get_repo_map("/other", "fp")

    warnings = [r for r in caplog.records if "Redis is not available" in r.message]
    assert len(warnings) == 1, "a down cache must not spam the log on every call"


async def test_an_unavailable_cache_stops_being_retried(fake_redis):
    """
    Paying the connect timeout on every call would make a missing Redis slower
    than no cache at all, so after the first failure no client is handed out
    and later calls short-circuit without touching the server.
    """
    fake_redis.broken = True
    await cache.get_repo_map("/proj", "fp")

    assert cache._redis() is None
    assert await cache.get_repo_map("/proj", "fp") is None


async def test_reset_clears_the_unavailable_flag(fake_redis):
    fake_redis.broken = True
    await cache.get_repo_map("/proj", "fp")
    assert cache._unavailable is True

    await cache.reset()
    assert cache._unavailable is False


async def test_a_write_failure_is_absorbed(monkeypatch):
    """The cache is an accelerator: a server that fails mid-write changes nothing."""

    class BrokenClient(FakeRedis):
        async def set(self, key, value, ex=None):
            raise RedisError("gone mid-write")

    monkeypatch.setattr(cache, "_client", BrokenClient())
    await cache.set_repo_map("/proj", "fp", "map")  # must not raise
