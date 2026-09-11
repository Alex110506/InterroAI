"""
OpenAI text-embedding-3-small batch client (Section 3C of architecture spec).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from core.llm import EMBED_TIMEOUT, embed_batch, get_client

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100


async def embed_texts(
    texts: list[str],
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[list[float]]:
    """
    Embed *texts* in batches using text-embedding-3-small.

    Calls ``on_progress(embedded_so_far, total)`` after each batch if provided.

    Raises:
        MissingAPIKeyError: no API key is stored in the keychain.
        openai.APIError: the provider was still failing after the retry budget
            in `core.llm` was exhausted. Batches already embedded are lost —
            the caller is expected to restart indexing rather than resume.
    """
    client = get_client(EMBED_TIMEOUT)
    result: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = await embed_batch(client, model=_MODEL, input=batch)
        result.extend(item.embedding for item in response.data)
        if on_progress:
            await on_progress(len(result), total)

    return result
