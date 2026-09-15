"""
The runtime's model gateway, talking to the real Cloud API app in-process.

Only OpenAI is faked, on the cloud side, so these tests pin down the wire format
itself: whatever `/llm/chat` sends, the runtime can rebuild. The last test
follows a refusal all the way to the chat event the app receives.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import openai
import pytest
from fakes.models import MODEL, FakeGateway, FakeStream, chat_chunks
from fakes.usage import InMemoryUsageMeter
from openai.types.chat import ChatCompletion

from agents.session import ChatSession
from cloud.api.main import create_app
from cloud.api.services import Limits, Services
from cloud.api.tokens import TokenSigner
from cloud.db.usage import Quota
from core.errors import CloudError, QuotaExceededError, UpstreamLLMError
from core.remote.gateway import RemoteModelGateway
from core.remote.session import CloudSession

CLOUD = "http://cloud.test"
MESSAGES = [{"role": "user", "content": "Say hello"}]
_ROOMY_QUOTA = Quota(requests=100, tokens=100_000)


def _gateway(quota: Quota = _ROOMY_QUOTA) -> tuple[RemoteModelGateway, FakeGateway]:
    signer = TokenSigner("g" * 32, issuer=CLOUD, access_ttl=timedelta(minutes=15))
    upstream = FakeGateway()
    services = Services(
        signin=None,
        signer=signer,
        accounts=None,
        projects=None,
        usage=InMemoryUsageMeter(),
        gateway=upstream,
        limits=Limits(quota=quota),
    )
    session = CloudSession(CLOUD, transport=httpx.ASGITransport(app=create_app(services)))
    session.adopt_tokens(
        access_token=signer.access_token("user-1", "octocat"),
        refresh_token="not-used-here",
        expires_in=900,
    )
    return RemoteModelGateway(session), upstream


@pytest.fixture
def world(fake_keyring):
    return _gateway()


async def test_a_completion_comes_back_as_openais_own_object(world):
    gateway, upstream = world

    completion = await gateway.chat(model=MODEL, messages=MESSAGES)

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Hello"
    assert completion.usage.total_tokens == 15
    assert upstream.requests == [{"model": MODEL, "messages": MESSAGES}]


async def test_a_stream_yields_only_chunks_the_agents_can_index(world):
    gateway, upstream = world

    stream = await gateway.chat_stream(model=MODEL, messages=MESSAGES)
    chunks = [chunk async for chunk in stream]

    assert all(chunk.choices for chunk in chunks), "the usage-only chunk must not reach the agents"
    assert "".join(chunk.choices[0].delta.content for chunk in chunks) == "Hello"
    assert upstream.requests[0]["stream_options"] == {"include_usage": True}


async def test_a_stream_cut_off_upstream_raises_after_what_arrived(world):
    gateway, upstream = world
    upstream.stream = FakeStream(
        chat_chunks("Hel")[:1],
        failure=openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
    )

    stream = await gateway.chat_stream(model=MODEL, messages=MESSAGES)
    received = []
    with pytest.raises(UpstreamLLMError):
        async for chunk in stream:
            received.append(chunk.choices[0].delta.content)

    assert received == ["Hel"]


async def test_a_quota_refusal_raises_before_any_stream_starts(fake_keyring):
    gateway, upstream = _gateway(Quota(requests=0, tokens=100_000))

    with pytest.raises(QuotaExceededError):
        await gateway.chat_stream(model=MODEL, messages=MESSAGES)
    with pytest.raises(QuotaExceededError):
        await gateway.chat(model=MODEL, messages=MESSAGES)

    assert upstream.requests == []


async def test_a_model_the_cloud_does_not_offer_is_refused_with_its_code(world):
    gateway, _ = world

    with pytest.raises(CloudError) as raised:
        await gateway.chat(model="gpt-4o", messages=MESSAGES)

    assert raised.value.code == "model_not_allowed"


async def test_a_refusal_reaches_the_app_as_an_error_event_with_its_code(tmp_path, fake_keyring):
    gateway, _ = _gateway(Quota(requests=0, tokens=100_000))
    session = ChatSession(
        project_path=str(tmp_path), project_index={}, model=None, history=None, gateway=gateway
    )

    events = [event async for event in session.start("Explain the auth flow")]

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "quota_exceeded"
