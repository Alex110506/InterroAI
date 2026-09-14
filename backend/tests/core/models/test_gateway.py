"""
The default ModelGateway: straight to OpenAI, through `core.models.llm`'s policy.

Retries and timeouts themselves are tested in `test_llm.py`. What is left here
is that the gateway routes through that policy rather than around it.
"""
from __future__ import annotations

import pytest

import core.models.gateway as gateway_module
from core.errors import MissingAPIKeyError
from core.models.gateway import FAST_TIMEOUT, LONG_TIMEOUT, OpenAIGateway


@pytest.fixture
def recorded(monkeypatch):
    calls: dict = {}

    def fake_get_client(timeout):
        calls["timeout"] = timeout
        return "client"

    async def fake_completion(client, **kwargs):
        calls["completion"] = (client, kwargs)
        return "a completion"

    async def fake_stream(client, **kwargs):
        calls["stream"] = (client, kwargs)
        return "a stream"

    monkeypatch.setattr(gateway_module, "get_client", fake_get_client)
    monkeypatch.setattr(gateway_module, "chat_completion", fake_completion)
    monkeypatch.setattr(gateway_module, "chat_stream", fake_stream)
    return calls


async def test_chat_goes_through_the_retrying_wrapper(recorded):
    result = await OpenAIGateway().chat(model="m", messages=[])
    assert result == "a completion"
    assert recorded["completion"] == ("client", {"model": "m", "messages": []})


async def test_chat_stream_returns_the_opened_stream(recorded):
    assert await OpenAIGateway().chat_stream(model="m", messages=[]) == "a stream"
    assert recorded["stream"] == ("client", {"model": "m", "messages": []})


async def test_the_timeout_it_is_given_configures_the_client(recorded):
    await OpenAIGateway().chat(timeout=FAST_TIMEOUT, model="m", messages=[])
    assert recorded["timeout"] is FAST_TIMEOUT


async def test_the_default_timeout_is_the_long_one(recorded):
    await OpenAIGateway().chat(model="m", messages=[])
    assert recorded["timeout"] is LONG_TIMEOUT


async def test_the_timeout_is_never_sent_as_a_request_field(recorded):
    """It configures the client; inside the request body OpenAI would reject it."""
    await OpenAIGateway().chat(timeout=FAST_TIMEOUT, model="m", messages=[])
    assert "timeout" not in recorded["completion"][1]


async def test_a_missing_key_surfaces_as_the_typed_error(without_api_key):
    with pytest.raises(MissingAPIKeyError):
        await OpenAIGateway().chat(model="m", messages=[])
