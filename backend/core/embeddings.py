"""
OpenAI text-embedding-3-small batch client (Section 3C of architecture spec).

`embed_batches` is the form the indexer uses, and it is a generator for two
reasons:

  * **Partial progress survives a failure.** Its caller persists each batch as
    it arrives, so a project that dies on batch 59 of 60 keeps the 58 batches
    already paid for. The previous version accumulated every vector in memory
    and stored them in one call at the very end, which meant any late failure
    threw away the whole run's API spend.
  * **Progress needs no callback.** The caller is itself a generator; when this
    one yields, progress can simply be re-yielded. An earlier version bridged a
    progress callback to a generator with an `asyncio.Queue`, which is now
    unnecessary.

One bad chunk does not sink the run. A batch that fails with an *item* error
(see `core.llm.ITEM_ERRORS` — a chunk over the token limit, say) is bisected to
find the offending items, which are reported as failures and skipped while the
rest of the batch is kept. Anything else — the provider being down, a bad key —
is environmental and propagates, because skipping items one at a time through a
real outage would quietly produce an empty index.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from core import cache
from core.llm import EMBED_TIMEOUT, ITEM_ERRORS, embed_batch, get_client

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100


@dataclass(frozen=True)
class EmbeddedBatch:
    """
    One batch's worth of finished embeddings, ready to persist.

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
) -> AsyncIterator[EmbeddedBatch]:
    """
    Embed *texts*, yielding each batch as it completes.

    Cached vectors are served without touching the API — identical text always
    has an identical embedding, so a moved or renamed file costs nothing.

    Raises:
        MissingAPIKeyError: no API key is stored in the keychain.
        openai.APIError: the provider failed in a way that is not attributable
            to a single input (see the module docstring).
    """
    if not texts:
        return

    client = get_client(EMBED_TIMEOUT)

    for start in range(0, len(texts), _BATCH_SIZE):
        positions = list(range(start, min(start + _BATCH_SIZE, len(texts))))

        # Identical text inside one batch shares a key, so group by key and
        # embed each distinct text once.
        by_key: dict[str, list[int]] = {}
        for position in positions:
            by_key.setdefault(cache.vector_key(texts[position], model), []).append(position)

        cached = await cache.get_vectors(list(by_key))
        missing = [key for key in by_key if key not in cached]

        fresh: dict[str, list[float]] = {}
        failures: dict[int, BaseException] = {}
        if missing:
            fresh, failed_keys = await _embed_keys(client, model, missing, texts, by_key)
            await cache.set_vectors(fresh)
            for key, exc in failed_keys.items():
                for position in by_key[key]:
                    failures[position] = exc

        indices: list[int] = []
        vectors: list[list[float]] = []
        from_cache = 0
        for key, key_positions in by_key.items():
            vector = cached.get(key)
            if vector is None:
                vector = fresh.get(key)
            else:
                from_cache += len(key_positions)
            if vector is None:
                continue
            for position in key_positions:
                indices.append(position)
                vectors.append(vector)

        # Sorted so a consumer storing batch-by-batch sees a stable order.
        order = sorted(range(len(indices)), key=lambda i: indices[i])
        yield EmbeddedBatch(
            indices=[indices[i] for i in order],
            vectors=[vectors[i] for i in order],
            from_cache=from_cache,
            failures=failures,
        )


async def _embed_keys(
    client,
    model: str,
    keys: list[str],
    texts: list[str],
    by_key: dict[str, list[int]],
) -> tuple[dict[str, list[float]], dict[str, BaseException]]:
    """Embed the text behind each of *keys*, isolating any item that fails."""
    inputs = [texts[by_key[key][0]] for key in keys]
    vectors, failed = await _embed_isolating(client, model, keys, inputs)
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


async def embed_texts(texts: list[str], *, model: str = _MODEL) -> list[list[float]]:
    """
    Embed *texts* and return every vector, in order.

    For callers that cannot proceed on partial results — embedding a single
    search query, say. A text that could not be embedded raises rather than
    leaving a hole in the returned list.

    Raises:
        MissingAPIKeyError: no API key is stored in the keychain.
        openai.APIError: the provider failed, or one of *texts* could not be
            embedded at all.
    """
    vectors: list[list[float] | None] = [None] * len(texts)

    async for batch in embed_batches(texts, model=model):
        if batch.failures:
            raise next(iter(batch.failures.values()))
        for position, vector in zip(batch.indices, batch.vectors, strict=True):
            vectors[position] = vector

    return [v for v in vectors if v is not None]
