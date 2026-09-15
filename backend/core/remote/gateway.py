"""
`ModelGateway` over the Cloud API's `POST /llm/chat`.

The agents cannot tell it apart from `OpenAIGateway`. Requests go out in
OpenAI's shape, and what comes back is rebuilt into OpenAI's own
`ChatCompletion` and `ChatCompletionChunk` objects. The API holds the key and
enforces the quotas.

One difference is smoothed over here. The API asks OpenAI to end every stream
with a usage chunk that has no choices, so it can count tokens. The agents read
`chunk.choices[0]`, as they safely may on a stream they opened themselves, so
those chunks are dropped here and never reach them.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from core.errors import UpstreamLLMError
from core.models.gateway import LONG_TIMEOUT
from core.remote.session import CloudSession
from core.remote.sse import read_events

#: The API spends the caller's whole budget on OpenAI itself, so the runtime
#: allows a little more than that. Giving up at the same moment would produce a
#: dropped connection here — reported as an unreachable cloud — instead of the
#: API's own answer, which says the model took too long. The total stays inside
#: Container Apps' 240-second limit on any one request.
_MARGIN_SECONDS = 30.0


def cloud_timeout(timeout: httpx.Timeout) -> httpx.Timeout:
    """*timeout*, plus the margin the extra hop through the Cloud API needs."""
    read = (timeout.read or LONG_TIMEOUT.read) + _MARGIN_SECONDS
    return httpx.Timeout(read, connect=timeout.connect)


class RemoteModelGateway:
    def __init__(self, session: CloudSession) -> None:
        self._session = session

    async def chat(
        self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any
    ) -> ChatCompletion:
        response = await self._session.request(
            "POST", "/llm/chat", json=request, timeout=cloud_timeout(timeout)
        )
        return ChatCompletion.model_validate(response.json())

    async def chat_stream(
        self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        # Opened here, in the await, so a refusal (quota, model) raises before
        # the caller starts iterating, as it would from OpenAI directly.
        exits = contextlib.AsyncExitStack()
        try:
            response = await exits.enter_async_context(
                self._session.stream(
                    "POST",
                    "/llm/chat",
                    json={**request, "stream": True},
                    timeout=cloud_timeout(timeout),
                )
            )
        except BaseException:
            await exits.aclose()
            raise
        return _chunks(response, exits)


async def _chunks(
    response: httpx.Response, exits: contextlib.AsyncExitStack
) -> AsyncIterator[ChatCompletionChunk]:
    try:
        async for event in read_events(response.aiter_lines()):
            if event.event == "error":
                raise UpstreamLLMError(_message(event.data))
            if event.data == "[DONE]":
                return
            chunk = ChatCompletionChunk.model_validate_json(event.data)
            if chunk.choices:
                yield chunk
        raise UpstreamLLMError("The model's answer was cut off before it finished.")
    except httpx.TransportError as exc:
        raise UpstreamLLMError(
            "The connection to the InterroAI cloud dropped in the middle of an answer."
        ) from exc
    finally:
        await exits.aclose()


def _message(data: str) -> str:
    try:
        return str(json.loads(data)["message"])
    except (ValueError, KeyError, TypeError):
        return "The model stopped responding part-way."
