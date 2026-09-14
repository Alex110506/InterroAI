"""
ModelGateway — the one door every chat-model call goes through.

`OpenAIGateway` opens straight onto OpenAI with the key from the OS keychain.
The cloud build swaps in a gateway that calls the Web API instead, which holds
the platform key and enforces per-user quotas; nothing in `agents/` should be
able to tell the difference, which is why the agents take a `ModelGateway` and
never import `core.models.llm` themselves.

Requests and responses are OpenAI's own chat-completion shapes, on purpose. A
schema of our own would mean translating tool calls, streaming deltas and
reasoning-model quirks in both directions, in two places, indefinitely. Passing
OpenAI's schema through keeps the cloud gateway a thin authenticated proxy and
leaves the agent code exactly as it is.

Embeddings are deliberately not on this interface. The runtime never embeds
anything: indexing hands chunks to a `SemanticIndex`, and a search hands it a
query string. Where the vectors come from is the index's concern — in the cloud,
the worker and the search endpoint, which sit next to the key.
"""
from __future__ import annotations

from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from core.models.llm import FAST_TIMEOUT, LONG_TIMEOUT, chat_completion, chat_stream, get_client

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

# Re-exported so callers pick a timeout class without reaching into `core.models.llm`.
__all__ = ["FAST_TIMEOUT", "LONG_TIMEOUT", "ModelGateway", "OpenAIGateway"]


class ModelGateway(Protocol):
    async def chat(
        self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any
    ) -> ChatCompletion:
        """One chat completion. *request* is OpenAI's `chat.completions.create` body."""
        ...

    async def chat_stream(
        self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any
    ) -> AsyncIterable[ChatCompletionChunk]:
        """
        Open a streaming chat completion and return the stream.

        Awaiting this establishes the stream; iterating it yields the chunks.
        Only establishment may be retried — once a token has reached the caller
        a retry would replay it.
        """
        ...


class OpenAIGateway:
    """Direct to OpenAI, with the key stored in the OS keychain."""

    async def chat(self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any):
        return await chat_completion(get_client(timeout), **request)

    async def chat_stream(self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any):
        return await chat_stream(get_client(timeout), **request)
