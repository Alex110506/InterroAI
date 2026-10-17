"""
ModelGateway — the one door every chat-model call goes through.

The desktop runtime's gateway (`core/remote/gateway.py`) calls the Cloud API,
which holds the platform key and enforces per-user quotas. The Cloud API's own
(`cloud/llm/gateway.py`, `OpenAIGateway`) opens straight onto OpenAI. Nothing in
`agents/` can tell the difference, which is why the agents take a `ModelGateway`
and never import an OpenAI client themselves.

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

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

# ── Timeouts ─────────────────────────────────────────────────────────────────
# Callers pick one per call, and both sides of the port read them: the agents
# pass one in, and the Cloud API waits the same budget on OpenAI. `connect` is
# short everywhere: failing to reach the API is immediately obvious, so there is
# no reason to wait on it. The read budget is what varies. Embedding batches
# have their own, beside the embedding client (`cloud/llm/openai_client.py`).

#: Intent classification — small prompts, must feel instant.
FAST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

#: Coder planning and tool rounds against a reasoning model.
LONG_TIMEOUT = httpx.Timeout(180.0, connect=5.0)


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
