"""
Central OpenAI client construction, timeout policy and retry policy.

Every OpenAI call in the app goes through here so that timeouts and backoff are
defined in exactly one place instead of being re-derived (or forgotten) at each
call site.

Two things worth knowing about the defaults this replaces:

  * The OpenAI SDK's default timeout is **600 seconds**. For a desktop UI that
    means one hung request pins a WebSocket for ten minutes with no feedback.
    We set per-call-class timeouts sized to what each call actually does.

  * The SDK *does* retry twice by default, but with its own backoff curve. We
    set ``max_retries=0`` and let tenacity own the policy outright — otherwise
    the two compound (4 tenacity attempts x 2 SDK retries = 8 real requests)
    and the effective timeout becomes impossible to reason about.

Streaming note: `chat_stream` retries only the *establishment* of the stream.
Once tokens have been yielded to the caller a retry would replay them, so
mid-stream failures propagate instead.
"""
from __future__ import annotations

import logging

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core.errors import MissingAPIKeyError

logger = logging.getLogger(__name__)

# ── Timeouts ─────────────────────────────────────────────────────────────────
# `connect` is short everywhere: failing to reach the API is immediately
# obvious, so there is no reason to wait on it. The read budget is what varies.

#: Intent classification — small prompts, must feel instant.
FAST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

#: Embedding batches — up to 100 inputs per request.
EMBED_TIMEOUT = httpx.Timeout(60.0, connect=5.0)

#: Coder planning and tool rounds against a reasoning model.
LONG_TIMEOUT = httpx.Timeout(180.0, connect=5.0)

# ── Retry policy ─────────────────────────────────────────────────────────────
#: Failures that are worth retrying: the request never produced a usable answer
#: and the same request may well succeed shortly. Everything else (auth errors,
#: malformed requests, content filtering) is deterministic — retrying only
#: wastes the user's time and money, so it propagates on the first attempt.
TRANSIENT_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

#: Failures that belong to one specific *input* rather than to the environment:
#: a chunk over the token limit, content the provider refuses. A batch request
#: carries many inputs, so one of these means the batch must be split to find
#: the offending item — not that the run is doomed.
#:
#: Deliberately a whitelist. An unrecognised error is treated as environmental
#: and aborts the run, because the alternative — skipping items one by one
#: through an auth failure — would quietly produce an empty index and report
#: success.
ITEM_ERRORS = (
    BadRequestError,
    UnprocessableEntityError,
)

_MAX_ATTEMPTS = 4


def _worth_retrying_a_chat(exc: BaseException) -> bool:
    """
    The transient failures, minus timeouts.

    A chat call that timed out is not worth repeating: the same long prompt
    takes the same long time, someone is waiting on it, and through the cloud
    the request is cancelled at the ingress before a second attempt could
    finish. Embeddings still retry timeouts — nobody waits on the worker, and a
    batch that does land is cached.

    `APITimeoutError` subclasses `APIConnectionError`, so it has to be ruled out
    explicitly rather than merely left out of the tuple.
    """
    return isinstance(exc, TRANSIENT_ERRORS) and not isinstance(exc, APITimeoutError)


def _retrying(condition):
    return retry(
        retry=condition,
        wait=wait_exponential_jitter(initial=1.0, max=20.0),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


llm_retry = _retrying(retry_if_exception_type(TRANSIENT_ERRORS))
chat_retry = _retrying(retry_if_exception(_worth_retrying_a_chat))

# ── Client construction ──────────────────────────────────────────────────────
# Each AsyncOpenAI owns an httpx.AsyncClient with its own connection pool. The
# previous code built one per call and never closed it, leaking a pool every
# time. Cache by (key, timeout) so connections are reused and a rotated key
# still produces a fresh client.
_clients: dict[tuple[str, float | None, float | None], AsyncOpenAI] = {}

#: The platform's key, set by the process that holds it: the cloud API and the
#: worker, from their settings. Nothing else reaches OpenAI — the desktop
#: runtime goes through the API's LLM gateway and never calls it directly.
_configured_key: str | None = None


def use_api_key(key: str | None) -> None:
    """
    Use *key* for every client from now on.

    For the cloud processes, whose key comes from settings — in Azure, from Key
    Vault through an environment variable. `None` clears it, leaving no key.
    """
    global _configured_key
    _configured_key = key or None
    _clients.clear()


def get_client(timeout: httpx.Timeout) -> AsyncOpenAI:
    """
    Return a client for the configured key.

    Raises:
        MissingAPIKeyError: if no key has been configured, which means the
            process holding the platform key started without one. This is an
            expected condition, not a bug — callers should surface it verbatim
            rather than treating it as a generic failure.
    """
    key = _configured_key
    if not key:
        raise MissingAPIKeyError()

    cache_key = (key, timeout.read, timeout.connect)
    client = _clients.get(cache_key)
    if client is None:
        client = AsyncOpenAI(api_key=key, timeout=timeout, max_retries=0)
        _clients[cache_key] = client
    return client


# ── Retrying call wrappers ───────────────────────────────────────────────────


@chat_retry
async def chat_completion(client: AsyncOpenAI, **kwargs):
    """A non-streaming chat completion, retried on transient failures but not on a timeout."""
    return await client.chat.completions.create(**kwargs)


@chat_retry
async def chat_stream(client: AsyncOpenAI, **kwargs):
    """
    Open a streaming chat completion.

    Only the handshake is retried — see the module docstring. The returned
    stream is consumed by the caller, and a failure mid-iteration propagates.
    """
    return await client.chat.completions.create(stream=True, **kwargs)


@llm_retry
async def embed_batch(client: AsyncOpenAI, **kwargs):
    """One embedding batch, retried on transient failures."""
    return await client.embeddings.create(**kwargs)
