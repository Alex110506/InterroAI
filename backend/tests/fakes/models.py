"""
A stand-in for the model gateway that answers in OpenAI's own response types.

Shared by the Cloud API's `/llm/chat` tests and the runtime's remote gateway
tests, so both sides are checked against the same objects.
"""
from __future__ import annotations

from openai.types.chat import ChatCompletion, ChatCompletionChunk

MODEL = "gpt-5.6-sol"


def _usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def chat_completion(
    text: str = "Hello", *, prompt: int = 12, completion: int = 3
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": _usage(prompt, completion),
        }
    )


def chat_chunk(**fields) -> ChatCompletionChunk:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    return ChatCompletionChunk.model_validate(base | fields)


def chat_chunks(*texts: str, prompt: int = 12, completion: int = 3) -> list[ChatCompletionChunk]:
    """One chunk per text, then the usage-only chunk `include_usage` asks OpenAI for."""
    deltas = [
        chat_chunk(choices=[{"index": 0, "delta": {"content": text}, "finish_reason": None}])
        for text in texts
    ]
    return [*deltas, chat_chunk(choices=[], usage=_usage(prompt, completion))]


class FakeStream:
    def __init__(self, chunks, *, failure: Exception | None = None) -> None:
        self._chunks = chunks
        self._failure = failure
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._failure is not None:
            raise self._failure

    async def close(self) -> None:
        self.closed = True


class FakeGateway:
    """Records every request; answers with `completion` or `stream`, or raises `failure`."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.completion = chat_completion()
        self.stream = FakeStream(chat_chunks("Hel", "lo"))
        self.failure: Exception | None = None

    async def chat(self, *, timeout, **request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.completion

    async def chat_stream(self, *, timeout, **request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.stream
