"""
The local `EmbeddingCache`: the optional Redis in `core/local/cache.py`.

It inherits that module's rule — an accelerator, never a dependency. With no
Redis running, every lookup misses and every write is dropped, so indexing
simply pays again for embeddings it could have reused.
"""
from __future__ import annotations

from core.local import cache


class RedisEmbeddingCache:
    async def get(self, model: str, digests: list[str]) -> dict[str, list[float]]:
        keys = {cache.vector_key_for(model, digest): digest for digest in digests}
        found = await cache.get_vectors(list(keys))
        return {keys[key]: vector for key, vector in found.items()}

    async def put(self, model: str, vectors: dict[str, list[float]]) -> None:
        await cache.set_vectors(
            {cache.vector_key_for(model, digest): vector for digest, vector in vectors.items()}
        )
