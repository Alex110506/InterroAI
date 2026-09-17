"""
Timeout and retry policy — the reliability boundary around every OpenAI call.

Only the cloud API and the worker reach OpenAI, each with the platform key from
its settings; `use_api_key` is how that key arrives. Retry tests use
`retry_with(wait=wait_none())` so the real exponential backoff is exercised for
*shape* (attempt counts, which exceptions requalify) without making the suite
sleep.
"""
from __future__ import annotations

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import wait_none

import core.models.llm as llm
from core.errors import MissingAPIKeyError

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status):
    return cls("boom", response=httpx.Response(status, request=REQUEST), body=None)


def _instant(fn):
    """The same retry policy with the sleeping removed."""
    return fn.retry_with(wait=wait_none())


# ── Client construction ──────────────────────────────────────────────────────


def test_no_configured_key_raises_a_typed_error():
    """A cloud process started without the platform key — expected, not a defect."""
    llm.use_api_key(None)
    with pytest.raises(MissingAPIKeyError):
        llm.get_client(llm.FAST_TIMEOUT)


def test_an_empty_key_is_treated_as_missing():
    llm.use_api_key("")
    with pytest.raises(MissingAPIKeyError):
        llm.get_client(llm.FAST_TIMEOUT)


def test_the_configured_key_is_the_one_used():
    llm.use_api_key("sk-platform")
    assert llm.get_client(llm.FAST_TIMEOUT).api_key == "sk-platform"


def test_clearing_the_key_leaves_none():
    llm.use_api_key("sk-platform")
    llm.use_api_key(None)
    with pytest.raises(MissingAPIKeyError):
        llm.get_client(llm.FAST_TIMEOUT)


def test_sdk_retries_are_disabled(with_api_key):
    """
    tenacity owns the retry policy. Leaving the SDK's default of 2 would
    compound to 4x2 = 8 real requests with two different backoff curves.
    """
    assert llm.get_client(llm.FAST_TIMEOUT).max_retries == 0


def test_requested_timeout_is_applied(with_api_key):
    assert llm.get_client(llm.LONG_TIMEOUT).timeout == llm.LONG_TIMEOUT


def test_clients_are_cached_per_timeout(with_api_key):
    """Each client owns an httpx pool; rebuilding per call leaked one each time."""
    first = llm.get_client(llm.FAST_TIMEOUT)
    assert llm.get_client(llm.FAST_TIMEOUT) is first
    assert llm.get_client(llm.LONG_TIMEOUT) is not first


def test_rotating_the_key_yields_a_fresh_client():
    llm.use_api_key("sk-old")
    old = llm.get_client(llm.FAST_TIMEOUT)
    llm.use_api_key("sk-new")
    assert llm.get_client(llm.FAST_TIMEOUT) is not old


# ── Timeout policy ───────────────────────────────────────────────────────────


def test_every_timeout_is_far_below_the_sdk_default():
    """The SDK default is 600s — long enough to pin a WebSocket for 10 minutes."""
    for timeout in (llm.FAST_TIMEOUT, llm.EMBED_TIMEOUT, llm.LONG_TIMEOUT):
        assert timeout.read < 600
        assert timeout.connect == 5.0


def test_timeouts_are_ordered_by_expected_work():
    assert llm.FAST_TIMEOUT.read < llm.EMBED_TIMEOUT.read < llm.LONG_TIMEOUT.read


# ── Which failures requalify for a retry ─────────────────────────────────────


def test_transient_set_covers_the_recoverable_failures():
    assert set(llm.TRANSIENT_ERRORS) == {
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    }


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=REQUEST),
        lambda: _status_error(RateLimitError, 429),
        lambda: _status_error(InternalServerError, 500),
    ],
)
async def test_transient_failures_are_retried_then_succeed(fake_client, error):
    exc = error() if callable(error) else error
    client = fake_client([exc, exc, "recovered"])
    assert await _instant(llm.chat_completion)(client, model="m") == "recovered"
    assert len(client.calls) == 3


async def test_retries_give_up_after_the_attempt_budget(fake_client):
    """Four attempts total, then the original error is re-raised (reraise=True)."""
    failures = [APITimeoutError(request=REQUEST) for _ in range(10)]
    client = fake_client(failures)
    with pytest.raises(APITimeoutError):
        await _instant(llm.embed_batch)(client, model="e", input=["x"])
    assert len(client.calls) == 4


async def test_a_chat_that_timed_out_is_not_tried_again(fake_client):
    """
    The same long prompt takes the same long time, and someone is waiting on it.
    Through the cloud, the request is cancelled at the ingress before a second
    attempt could finish.
    """
    client = fake_client([APITimeoutError(request=REQUEST) for _ in range(4)])

    with pytest.raises(APITimeoutError):
        await _instant(llm.chat_completion)(client, model="m")

    assert len(client.calls) == 1


async def test_an_embedding_that_timed_out_is_tried_again(fake_client):
    """Nobody waits on the worker, and a batch that does land is cached."""
    client = fake_client([APITimeoutError(request=REQUEST), "vectors"])

    assert await _instant(llm.embed_batch)(client, model="e", input=["x"]) == "vectors"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: _status_error(AuthenticationError, 401),
        lambda: _status_error(BadRequestError, 400),
        lambda: ValueError("programming error"),
    ],
)
async def test_deterministic_failures_are_not_retried(fake_client, make_error):
    """Retrying a bad key or a malformed request only wastes time and money."""
    client = fake_client([make_error() for _ in range(5)])
    with pytest.raises(Exception):  # noqa: B017
        await _instant(llm.chat_completion)(client, model="m")
    assert len(client.calls) == 1


async def test_a_first_attempt_success_makes_no_extra_calls(fake_client):
    client = fake_client(["ok"])
    assert await llm.chat_completion(client, model="m") == "ok"
    assert len(client.calls) == 1


# ── Wrapper behaviour ────────────────────────────────────────────────────────


async def test_chat_completion_forwards_its_kwargs(fake_client):
    client = fake_client(["ok"])
    await llm.chat_completion(client, model="m", temperature=0.4, tools=[{"a": 1}])
    assert client.calls[0] == {"model": "m", "temperature": 0.4, "tools": [{"a": 1}]}


async def test_chat_stream_sets_stream_true(fake_client):
    """The caller must never have to remember the flag."""
    client = fake_client(["stream-object"])
    await llm.chat_stream(client, model="m")
    assert client.calls[0]["stream"] is True


async def test_chat_stream_retries_only_the_handshake(fake_client):
    """
    Establishing the stream is safe to retry because nothing has been yielded
    yet. (Mid-iteration failures are the caller's problem by design.)
    """
    client = fake_client([APIConnectionError(request=REQUEST), "stream-object"])
    assert await _instant(llm.chat_stream)(client, model="m") == "stream-object"
    assert len(client.calls) == 2


async def test_embed_batch_retries_transient_failures(fake_client):
    client = fake_client([_status_error(RateLimitError, 429), "vectors"])
    assert await _instant(llm.embed_batch)(client, model="e", input=["x"]) == "vectors"
    assert len(client.calls) == 2


async def test_retry_warns_before_sleeping(fake_client, caplog):
    """A silent retry hides a degrading provider; each one must be logged."""
    client = fake_client([APIConnectionError(request=REQUEST), "ok"])
    with caplog.at_level("WARNING", logger="core.models.llm"):
        await _instant(llm.chat_completion)(client, model="m")
    assert any("Retrying" in r.message for r in caplog.records)
