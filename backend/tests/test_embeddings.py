"""
Embedding batching. The OpenAI client is replaced wholesale, so no test here
makes a network call.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

import core.embeddings as embeddings
from core.embeddings import _BATCH_SIZE, _MODEL, EmbeddedBatch, embed_batches, embed_texts
from core.errors import MissingAPIKeyError


def _embedding_response(count: int, dim: int = 4):
    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * dim) for _ in range(count)])


class RecordingClient:
    """Returns one vector per input and records the batches it was given."""

    def __init__(self):
        self.batches: list[list[str]] = []

    async def create(self, **kwargs):
        self.batches.append(kwargs["input"])
        return _embedding_response(len(kwargs["input"]))


@pytest.fixture
def recording(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: client)

    async def passthrough(_client, **kwargs):
        return await client.create(**kwargs)

    monkeypatch.setattr(embeddings, "embed_batch", passthrough)
    return client


async def test_missing_key_propagates_as_a_typed_error(monkeypatch):
    """Indexing must report the real cause, not a generic failure."""
    def boom(timeout):
        raise MissingAPIKeyError()

    monkeypatch.setattr(embeddings, "get_client", boom)
    with pytest.raises(MissingAPIKeyError):
        await embed_texts(["anything"])


async def test_returns_one_vector_per_input(recording):
    assert len(await embed_texts(["a", "b", "c"])) == 3


async def test_empty_input_makes_no_request(recording):
    assert await embed_texts([]) == []
    assert recording.batches == []


async def test_a_small_list_is_a_single_request(recording):
    await embed_texts(["a", "b"])
    assert len(recording.batches) == 1


async def test_large_input_is_split_into_batches(recording):
    total = _BATCH_SIZE * 2 + 5
    await embed_texts([f"t{i}" for i in range(total)])
    assert [len(b) for b in recording.batches] == [_BATCH_SIZE, _BATCH_SIZE, 5]


async def test_no_batch_exceeds_the_limit(recording):
    await embed_texts([f"t{i}" for i in range(_BATCH_SIZE * 3)])
    assert all(len(b) <= _BATCH_SIZE for b in recording.batches)


async def test_input_order_is_preserved_across_batches(recording):
    texts = [f"t{i}" for i in range(_BATCH_SIZE + 10)]
    await embed_texts(texts)
    assert [t for batch in recording.batches for t in batch] == texts


async def test_uses_the_configured_model(monkeypatch):
    captured = {}

    async def capture(_client, **kwargs):
        captured.update(kwargs)
        return _embedding_response(len(kwargs["input"]))

    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())
    monkeypatch.setattr(embeddings, "embed_batch", capture)
    await embed_texts(["x"])
    assert captured["model"] == _MODEL


async def test_uses_the_embedding_timeout(monkeypatch):
    seen = {}

    def capture_timeout(timeout):
        seen["timeout"] = timeout
        return object()

    async def fake_batch(_client, **kwargs):
        return _embedding_response(len(kwargs["input"]))

    monkeypatch.setattr(embeddings, "get_client", capture_timeout)
    monkeypatch.setattr(embeddings, "embed_batch", fake_batch)
    await embed_texts(["x"])
    assert seen["timeout"] is embeddings.EMBED_TIMEOUT


async def test_provider_errors_propagate(monkeypatch):
    """
    After `core.llm` exhausts its retries the failure is real; indexing should
    surface it rather than persist a half-embedded index.
    """
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())

    async def always_fails(_client, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(embeddings, "embed_batch", always_fails)
    with pytest.raises(RuntimeError, match="provider down"):
        await embed_texts(["x"])


# ── Batch-at-a-time delivery ─────────────────────────────────────────────────


async def _collect(texts) -> list[EmbeddedBatch]:
    return [batch async for batch in embed_batches(texts)]


async def test_each_batch_is_yielded_as_it_completes(recording):
    """
    The caller persists per batch, so a late failure cannot discard what the
    earlier batches already paid for.
    """
    total = _BATCH_SIZE * 2 + 3
    batches = await _collect([f"t{i}" for i in range(total)])

    assert [len(b.indices) for b in batches] == [_BATCH_SIZE, _BATCH_SIZE, 3]
    assert [i for b in batches for i in b.indices] == list(range(total))


async def test_a_batch_carries_vectors_aligned_with_its_indices(recording):
    batches = await _collect(["a", "b"])
    assert len(batches) == 1
    assert batches[0].indices == [0, 1]
    assert len(batches[0].vectors) == 2


async def test_nothing_is_yielded_for_no_input(recording):
    assert await _collect([]) == []


# ── The cache ────────────────────────────────────────────────────────────────


async def test_a_cached_vector_is_not_embedded_again(recording):
    """Content addressed, so re-indexing an unchanged chunk costs nothing."""
    await embed_texts(["stable chunk"])
    assert len(recording.batches) == 1

    await embed_texts(["stable chunk"])
    assert len(recording.batches) == 1, "the second call must be served by the cache"


async def test_a_cache_hit_is_reported_as_such(recording):
    await embed_texts(["x"])
    batches = await _collect(["x"])
    assert batches[0].from_cache == 1


async def test_a_renamed_file_reuses_its_vectors(recording):
    """The point of hashing content rather than paths: moving code is free."""
    await embed_texts(["def login(): ...", "def logout(): ..."])
    calls = len(recording.batches)

    # Same chunks, different order — as a moved file would produce.
    await embed_texts(["def logout(): ...", "def login(): ..."])
    assert len(recording.batches) == calls


async def test_duplicate_text_in_one_batch_is_embedded_once(recording):
    await embed_texts(["same", "same", "different"])
    assert sorted(recording.batches[0]) == ["different", "same"]


async def test_a_broken_cache_does_not_stop_embedding(recording, fake_redis):
    """Redis is an accelerator; losing it costs speed, never correctness."""
    fake_redis.broken = True
    assert len(await embed_texts(["a", "b"])) == 2


# ── One bad chunk does not sink the run ──────────────────────────────────────


def _bad_request(message: str = "input too long") -> BadRequestError:
    return BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com")),
        body=None,
    )


async def test_a_bad_chunk_is_skipped_and_the_rest_survive(monkeypatch):
    """
    The reported bug: one chunk the provider refuses used to fail the whole
    project. The batch is bisected instead, and only the offender is dropped.
    """
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())

    async def refuse_the_poison(_client, **kwargs):
        if "poison" in kwargs["input"]:
            raise _bad_request()
        return _embedding_response(len(kwargs["input"]))

    monkeypatch.setattr(embeddings, "embed_batch", refuse_the_poison)
    batches = await _collect(["ok one", "poison", "ok two"])

    assert [len(b.indices) for b in batches] == [2], "the good chunks must still land"
    assert list(batches[0].failures) == [1]
    assert isinstance(batches[0].failures[1], BadRequestError)


async def test_a_skipped_chunk_is_not_cached(monkeypatch, fake_redis):
    """A failure must not be remembered as a result."""
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())

    async def always_refuses(_client, **kwargs):
        raise _bad_request()

    monkeypatch.setattr(embeddings, "embed_batch", always_refuses)
    await _collect(["poison"])
    assert fake_redis.store == {}


async def test_isolating_one_bad_chunk_does_not_cost_a_call_per_chunk(monkeypatch):
    """Bisection, not one-request-per-item: O(log n) extra calls, not O(n)."""
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())
    calls = 0

    async def refuse_the_poison(_client, **kwargs):
        nonlocal calls
        calls += 1
        if "poison" in kwargs["input"]:
            raise _bad_request()
        return _embedding_response(len(kwargs["input"]))

    monkeypatch.setattr(embeddings, "embed_batch", refuse_the_poison)
    texts = [f"t{i}" for i in range(64)]
    texts[30] = "poison"
    await _collect(texts)

    assert calls < 20, f"bisection should isolate one item in a few calls, took {calls}"


async def test_an_environmental_failure_still_aborts(monkeypatch):
    """
    Skipping items one by one through an outage would quietly produce an empty
    index and report success, so only *item* errors are recoverable.
    """
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())

    async def provider_down(_client, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    monkeypatch.setattr(embeddings, "embed_batch", provider_down)
    with pytest.raises(APIConnectionError):
        await _collect(["a", "b", "c"])


async def test_embed_texts_refuses_to_return_a_hole(monkeypatch):
    """A single query caller cannot proceed on a partial result."""
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())

    async def always_refuses(_client, **kwargs):
        raise _bad_request()

    monkeypatch.setattr(embeddings, "embed_batch", always_refuses)
    with pytest.raises(BadRequestError):
        await embed_texts(["a query"])
