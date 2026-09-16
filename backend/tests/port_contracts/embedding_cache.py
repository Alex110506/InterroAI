"""
What every `EmbeddingCache` must do identically — Redis locally, Postgres in the cloud.

Subclass `EmbeddingCacheContract` and provide `cache` and `dimensions`.
"""
from __future__ import annotations

import pytest


class EmbeddingCacheContract:
    async def test_an_unknown_digest_is_a_miss(self, cache):
        assert await cache.get("model-a", ["never-stored"]) == {}

    async def test_a_stored_vector_comes_back_and_misses_stay_absent(self, cache, dimensions):
        vector = [0.25] * dimensions
        await cache.put("model-a", {"digest-1": vector})

        found = await cache.get("model-a", ["digest-1", "digest-2"])

        assert list(found) == ["digest-1"]
        # Stores may keep single precision; the vector must survive, not its bits.
        assert found["digest-1"] == pytest.approx(vector)

    async def test_models_never_share_vectors(self, cache, dimensions):
        """Two models give incompatible vectors for identical text."""
        await cache.put("model-a", {"digest-1": [0.5] * dimensions})
        assert await cache.get("model-b", ["digest-1"]) == {}

    async def test_writing_the_same_digest_twice_is_harmless(self, cache, dimensions):
        await cache.put("model-a", {"digest-1": [0.5] * dimensions})
        await cache.put("model-a", {"digest-1": [0.5] * dimensions})
        assert list(await cache.get("model-a", ["digest-1"])) == ["digest-1"]

    async def test_empty_requests_are_harmless(self, cache):
        assert await cache.get("model-a", []) == {}
        await cache.put("model-a", {})
