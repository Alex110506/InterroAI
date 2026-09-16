"""
OpenAI text-embedding-3-small batch client (Section 3C of architecture spec).

`embed_batches` is a generator that yields each batch as it completes, for two
reasons:

  * **Paid-for work survives a failure.** Given an `EmbeddingCache`, every batch
    is written to the cache the moment it lands, so a job that dies on batch 59
    of 60 re-embeds none of the first 58 when it runs again. (Without a cache —
    the local build with no Redis running — a failed job pays again.) The index
    itself is written once, atomically, after the last batch: see
    `core/index/indexer.py`.
  * **Progress needs no callback.** The caller is itself a generator; when this
    one yields, progress can simply be re-yielded.

One bad chunk does not sink the run. A batch that fails with an *item* error
(see `core.models.llm.ITEM_ERRORS` — a chunk over the token limit, say) is
bisected to find the offending items, which are reported as failures and
skipped while the rest of the batch is kept. Anything else — the provider being
down, a bad key — is environmental and propagates, because skipping items one at
a time through a real outage would quietly produce an empty index.

The cache is an accelerator in every build: one that fails to answer or to
store is logged and treated as a miss, never allowed to fail the embedding.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from core.index.ports import EmbeddingCache, content_digest
from core.models.llm import EMBED_TIMEOUT, ITEM_ERRORS, embed_batch, get_client

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100


@dataclass(frozen=True)
class EmbeddedBatch:
    """
    One batch's worth of finished embeddings.

    `indices` are positions in the list originally handed to `embed_batches`,
    and `vectors` is aligned with it. Positions that could not be embedded at
    all appear in `failures` instead, mapped to the error that stopped them.
    """

    indices: list[int]
    vectors: list[list[float]]
    from_cache: int = 0
    failures: dict[int, BaseException] = field(default_factory=dict)


async def embed_batches(
    texts: list[str],
    *,
    model: str = _MODEL,
    cache: EmbeddingCache | None = None,
) -> AsyncIterator[EmbeddedBatch]:
    """
    Embed *texts*, yielding each batch as it completes.

    Vectors already in *cache* are served without touching the API — identical
    text always has an identical embedding, so a moved or renamed file costs
    nothing — and every freshly embedded vector is written to it.

    Raises:
        MissingAPIKeyError: no API key is configured.
        openai.APIError: the provider failed in a way that is not attributable
            to a single input (see the module docstring).
    """
    if not texts:
        return

    client = get_client(EMBED_TIMEOUT)

    for start in range(0, len(texts), _BATCH_SIZE):
        positions = list(range(start, min(start + _BATCH_SIZE, len(texts))))

        # Identical text inside one batch shares a digest, so each distinct
        # text is embedded once.
        by_digest: dict[str, list[int]] = {}
        for position in positions:
            by_digest.setdefault(content_digest(texts[position]), []).append(position)

        cached = await _cache_get(cache, model, list(by_digest))
        missing = [digest for digest in by_digest if digest not in cached]

        fresh: dict[str, list[float]] = {}
        failures: dict[int, BaseException] = {}
        if missing:
            fresh, failed = await _embed_digests(client, model, missing, texts, by_digest)
            await _cache_put(cache, model, fresh)
            for digest, exc in failed.items():
                for position in by_digest[digest]:
                    failures[position] = exc

        indices: list[int] = []
        vectors: list[list[float]] = []
        from_cache = 0
        for digest, digest_positions in by_digest.items():
            vector = cached.get(digest)
            if vector is None:
                vector = fresh.get(digest)
            else:
                from_cache += len(digest_positions)
            if vector is None:
                continue
            for position in digest_positions:
                indices.append(position)
                vectors.append(vector)

        # Sorted so a consumer sees a stable order.
        order = sorted(range(len(indices)), key=lambda i: indices[i])
        yield EmbeddedBatch(
            indices=[indices[i] for i in order],
            vectors=[vectors[i] for i in order],
            from_cache=from_cache,
            failures=failures,
        )


async def _cache_get(
    cache: EmbeddingCache | None, model: str, digests: list[str]
) -> dict[str, list[float]]:
    if cache is None or not digests:
        return {}
    try:
        return await cache.get(model, digests)
    except Exception:  # noqa: BLE001
        logger.warning("The embedding cache could not be read; carrying on.", exc_info=True)
        return {}


async def _cache_put(
    cache: EmbeddingCache | None, model: str, vectors: dict[str, list[float]]
) -> None:
    if cache is None or not vectors:
        return
    try:
        await cache.put(model, vectors)
    except Exception:  # noqa: BLE001
        logger.warning("The embedding cache could not be written; carrying on.", exc_info=True)


async def _embed_digests(
    client,
    model: str,
    digests: list[str],
    texts: list[str],
    by_digest: dict[str, list[int]],
) -> tuple[dict[str, list[float]], dict[str, BaseException]]:
    """Embed the text behind each of *digests*, isolating any item that fails."""
    inputs = [texts[by_digest[digest][0]] for digest in digests]
    vectors, failed = await _embed_isolating(client, model, digests, inputs)
    if failed:
        logger.warning(
            "%d chunk(s) could not be embedded and were skipped: %s",
            len(failed),
            "; ".join(str(exc) for exc in list(failed.values())[:3]),
        )
    return vectors, failed


async def _embed_isolating(
    client,
    model: str,
    keys: list[str],
    inputs: list[str],
) -> tuple[dict[str, list[float]], dict[str, BaseException]]:
    """
    One request for the whole slice; on an item error, split and retry halves.

    Bisecting costs O(log n) extra requests to find one bad input, where
    retrying every item individually would cost O(n) — and on a provider
    outage, which is *not* an item error, we never get here at all.
    """
    try:
        response = await embed_batch(client, model=model, input=inputs)
        # One embedding per input is the API's contract; a mismatch would
        # silently pair a vector with the wrong chunk, so let it raise.
        return {
            key: item.embedding
            for key, item in zip(keys, response.data, strict=True)
        }, {}
    except ITEM_ERRORS as exc:
        if len(inputs) == 1:
            return {}, {keys[0]: exc}

    middle = len(inputs) // 2
    left_vectors, left_failed = await _embed_isolating(
        client, model, keys[:middle], inputs[:middle]
    )
    right_vectors, right_failed = await _embed_isolating(
        client, model, keys[middle:], inputs[middle:]
    )
    return {**left_vectors, **right_vectors}, {**left_failed, **right_failed}


async def embed_texts(
    texts: list[str],
    *,
    model: str = _MODEL,
    cache: EmbeddingCache | None = None,
) -> list[list[float]]:
    """
    Embed *texts* and return every vector, in order.

    For callers that cannot proceed on partial results — embedding a single
    search query, say. A text that could not be embedded raises rather than
    leaving a hole in the returned list.

    Raises:
        MissingAPIKeyError: no API key is configured.
        openai.APIError: the provider failed, or one of *texts* could not be
            embedded at all.
    """
    vectors: list[list[float] | None] = [None] * len(texts)

    async for batch in embed_batches(texts, model=model, cache=cache):
        if batch.failures:
            raise next(iter(batch.failures.values()))
        for position, vector in zip(batch.indices, batch.vectors, strict=True):
            vectors[position] = vector

    return [v for v in vectors if v is not None]
