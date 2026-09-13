"""
The Redis cache.

Two properties matter more than anything else here: that a cached value is the
same value that went in, and that *no* failure of the cache can fail the app.
The in-memory double from `conftest.py` stands in for the server, so these
tests exercise the real keying and packing without one.
"""
from __future__ import annotations

import pytest
from conftest import FakeRedis
from redis.exceptions import RedisError

import core.cache as cache

# ── Vectors ──────────────────────────────────────────────────────────────────


async def test_a_vector_survives_the_round_trip():
    key = cache.vector_key("some chunk", "text-embedding-3-small")
    await cache.set_vectors({key: [0.25, -0.5, 0.75]})

    assert await cache.get_vectors([key]) == pytest.approx({key: [0.25, -0.5, 0.75]})


async def test_an_uncached_key_is_simply_absent():
    assert await cache.get_vectors(["interroai:v1:vec:m:nope"]) == {}


async def test_asking_for_nothing_touches_no_server(fake_redis):
    assert await cache.get_vectors([]) == {}
    assert fake_redis.reads == 0


async def test_storing_nothing_touches_no_server(fake_redis):
    await cache.set_vectors({})
    assert fake_redis.sets == 0


async def test_identical_text_maps_to_one_key():
    model = "text-embedding-3-small"
    assert cache.vector_key("def f(): ...", model) == cache.vector_key("def f(): ...", model)


async def test_different_models_never_share_a_key():
    """
    Two models produce incompatible vectors for identical text; sharing a key
    would silently corrupt every search.
    """
    assert cache.vector_key("x", "text-embedding-3-small") != cache.vector_key("x", "other")


async def test_a_corrupt_value_reads_as_a_miss(fake_redis):
    """A truncated or foreign value must not crash the caller — it recomputes."""
    key = cache.vector_key("chunk", "m")
    fake_redis.store[key] = b"\x01\x02\x03"  # not a whole number of float32s

    assert await cache.get_vectors([key]) == {}


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


# ── Degradation ──────────────────────────────────────────────────────────────


async def test_reads_miss_when_redis_is_unreachable(fake_redis):
    fake_redis.broken = True
    assert await cache.get_vectors([cache.vector_key("x", "m")]) == {}
    assert await cache.get_repo_map("/proj", "fp") is None


async def test_writes_are_dropped_when_redis_is_unreachable(fake_redis):
    fake_redis.broken = True
    await cache.set_vectors({cache.vector_key("x", "m"): [0.1]})  # must not raise
    await cache.set_repo_map("/proj", "fp", "map")                # must not raise


async def test_the_first_failure_explains_itself_once(fake_redis, caplog):
    fake_redis.broken = True
    with caplog.at_level("WARNING", logger="core.cache"):
        await cache.get_repo_map("/proj", "fp")
        await cache.get_repo_map("/proj", "fp")
        await cache.get_vectors([cache.vector_key("x", "m")])

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


async def test_a_pipeline_failure_is_absorbed(monkeypatch):
    """Writes go through a pipeline, whose `execute` can fail on its own."""
    class BrokenPipeline:
        def set(self, *args, **kwargs):
            return self

        async def execute(self):
            raise RedisError("gone mid-write")

    client = FakeRedis()
    monkeypatch.setattr(client, "pipeline", lambda transaction=False: BrokenPipeline())
    monkeypatch.setattr(cache, "_client", client)

    await cache.set_vectors({cache.vector_key("x", "m"): [0.1]})  # must not raise
