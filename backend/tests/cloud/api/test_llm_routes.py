"""
`POST /llm/chat` over HTTP, on a fake model gateway and in-memory usage.

No request reaches OpenAI. The fake returns OpenAI's own response types, so the
tests also prove that what the API sends back can be rebuilt into them.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import openai
import pytest
from fakes.models import MODEL, FakeGateway, FakeStream, chat_chunks
from fakes.usage import InMemoryUsageMeter
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from cloud.api.main import create_app
from cloud.api.services import Limits, Services
from cloud.api.tokens import TokenSigner
from cloud.db.usage import Quota
from core.errors import MissingAPIKeyError

MESSAGES = [{"role": "user", "content": "Say hello"}]
USER_ID = "user-1"
_ROOMY_QUOTA = Quota(requests=100, tokens=100_000)


def _openai_error(cls, status: int):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls(
        "The model refused this request.",
        response=httpx.Response(status, request=request),
        body=None,
    )


@pytest.fixture
def api():
    return _api()


def _api(quota: Quota = _ROOMY_QUOTA):
    signer = TokenSigner("l" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))
    gateway, usage = FakeGateway(), InMemoryUsageMeter()
    services = Services(
        signin=None,
        signer=signer,
        accounts=None,
        projects=None,
        usage=usage,
        gateway=gateway,
        limits=Limits(quota=quota),
    )
    return SimpleNamespace(
        client=TestClient(create_app(services)),
        gateway=gateway,
        usage=usage,
        headers={"Authorization": f"Bearer {signer.access_token(USER_ID, 'Alex110506')}"},
    )


def _post(api, **body):
    return api.client.post(
        "/llm/chat", json={"model": MODEL, "messages": MESSAGES} | body, headers=api.headers
    )


def _data_lines(body: str) -> list[str]:
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


async def _today_totals(api):
    return await api.usage.totals(USER_ID, datetime.now(UTC).date())


# ── Completions ──────────────────────────────────────────────────────────────


async def test_a_completion_comes_back_as_openais_own_json(api):
    response = _post(api)

    assert response.status_code == 200
    rebuilt = ChatCompletion.model_validate(response.json())
    assert rebuilt.choices[0].message.content == "Hello"
    assert api.gateway.requests == [{"model": MODEL, "messages": MESSAGES}]
    totals = await _today_totals(api)
    assert (totals.requests, totals.prompt_tokens, totals.completion_tokens) == (1, 12, 3)


async def test_a_stream_is_relayed_as_openai_chunks_ending_in_done(api):
    response = _post(api, stream=True)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    *chunks, done = _data_lines(response.text)
    assert done == "[DONE]"
    rebuilt = [ChatCompletionChunk.model_validate_json(chunk) for chunk in chunks]
    assert "".join(c.choices[0].delta.content for c in rebuilt if c.choices) == "Hello"

    [request] = api.gateway.requests
    assert request["stream_options"] == {"include_usage": True}
    assert "stream" not in request
    assert api.gateway.stream.closed
    assert (await _today_totals(api)).completion_tokens == 3


async def test_a_stream_that_fails_part_way_ends_in_an_error_event_not_done(api):
    connection_lost = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    api.gateway.stream = FakeStream(chat_chunks("Hel")[:1], failure=connection_lost)

    response = _post(api, stream=True)

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "[DONE]" not in response.text
    assert json.loads(_data_lines(response.text)[-1])["code"] == "upstream_error"


# ── What is refused ──────────────────────────────────────────────────────────


async def test_only_the_agents_models_are_available(api):
    response = _post(api, model="gpt-4o")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "model_not_allowed"
    assert api.gateway.requests == []
    assert (await _today_totals(api)).requests == 0, "a refused request costs no allowance"


def test_reasoning_effort_is_forwarded(api):
    """
    The runtime sends "none" on every tools request, because these models reason
    by default and chat completions refuses tools alongside reasoning. The
    gateway refusing the parameter would break every tool round.
    """
    assert _post(api, reasoning_effort="none").status_code == 200
    assert api.gateway.requests[0]["reasoning_effort"] == "none"


@pytest.mark.parametrize("parameter", [{"n": 5}, {"store": True}, {"stream_options": {}}])
def test_parameters_outside_the_list_are_refused(api, parameter):
    response = _post(api, **parameter)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_parameter"
    assert api.gateway.requests == []


def test_messages_are_required(api):
    response = api.client.post(
        "/llm/chat", json={"model": MODEL, "messages": []}, headers=api.headers
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"


def test_chat_needs_a_signed_in_user(api):
    response = api.client.post("/llm/chat", json={"model": MODEL, "messages": MESSAGES})
    assert response.status_code == 401


# ── Quotas ───────────────────────────────────────────────────────────────────


def test_requests_past_the_daily_quota_are_429_until_midnight():
    api = _api(Quota(requests=1, tokens=100_000))

    assert _post(api).status_code == 200
    refused = _post(api)

    assert refused.status_code == 429
    assert refused.json()["detail"]["code"] == "quota_exceeded"
    assert 0 < int(refused.headers["retry-after"]) <= 24 * 3600 + 1


def test_tokens_past_the_daily_quota_refuse_the_next_request():
    api = _api(Quota(requests=100, tokens=10))

    assert _post(api).status_code == 200  # 15 tokens: the one response in flight may overshoot
    assert _post(api).status_code == 429


# ── Upstream failures ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (_openai_error(openai.BadRequestError, 400), 400, "invalid_request"),
        (MissingAPIKeyError(), 503, "llm_unavailable"),
        (_openai_error(openai.AuthenticationError, 401), 503, "llm_unavailable"),
        (_openai_error(openai.RateLimitError, 429), 503, "llm_unavailable"),
        (_openai_error(openai.InternalServerError, 500), 502, "upstream_error"),
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            ),
            504,
            "model_timeout",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_upstream_failures_become_codes_the_runtime_can_act_on(
    api, failure, status, code, stream
):
    api.gateway.failure = failure

    response = _post(api, stream=stream)

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
