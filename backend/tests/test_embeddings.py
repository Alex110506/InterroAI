"""
Embedding batching. The OpenAI client is replaced wholesale, so no test here
makes a network call.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.embeddings as embeddings
from core.embeddings import _BATCH_SIZE, _MODEL, embed_texts
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


async def test_progress_is_reported_after_each_batch(recording):
    seen: list[tuple[int, int]] = []

    async def on_progress(done, total):
        seen.append((done, total))

    total = _BATCH_SIZE * 2 + 3
    await embed_texts([f"t{i}" for i in range(total)], on_progress=on_progress)
    assert seen == [(_BATCH_SIZE, total), (_BATCH_SIZE * 2, total), (total, total)]


async def test_progress_is_optional(recording):
    await embed_texts(["a"])  # must not raise without a callback


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
