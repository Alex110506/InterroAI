"""The local EmbeddingCache over Redis — the in-memory `fake_redis` stands in for the server."""
from __future__ import annotations

import pytest
from port_contracts.embedding_cache import EmbeddingCacheContract

from core.index.adapters.redis_cache import RedisEmbeddingCache
from core.index.ports import content_digest
from core.local import cache as cache_module


class TestRedisEmbeddingCache(EmbeddingCacheContract):
    @pytest.fixture
    def cache(self):
        return RedisEmbeddingCache()

    @pytest.fixture
    def dimensions(self):
        return 8


async def test_vectors_cached_before_the_port_existed_are_still_found():
    """The key scheme did not change, so an existing Redis keeps paying off."""
    text = "def login(): ..."
    await cache_module.set_vectors({cache_module.vector_key(text, "model-a"): [0.5] * 4})

    found = await RedisEmbeddingCache().get("model-a", [content_digest(text)])

    assert found == {content_digest(text): pytest.approx([0.5] * 4)}


async def test_an_unreachable_redis_reads_as_a_miss(fake_redis):
    fake_redis.broken = True
    assert await RedisEmbeddingCache().get("model-a", ["digest"]) == {}
