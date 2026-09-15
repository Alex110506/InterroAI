"""
`POST /llm/chat`: the platform's OpenAI key, behind sign-in, allowlists and quotas.

The body is OpenAI's own `chat.completions.create` request. The response is
OpenAI's own completion, or its chunks as server-sent events, so the runtime can
rebuild exactly the objects the agents already handle. The API is a thin
authenticated proxy on purpose: it checks, counts and forwards, and knows nothing
about tool calls or reasoning models.

What it does enforce:

  * the model must be one the agents use;
  * only parameters from a fixed list pass through, so a request cannot ask
    for fifty completions (`n`) or have OpenAI keep the data (`store`);
  * every request is admitted against the user's daily request and token
    quotas, and the tokens it used are counted when it finishes. For streams,
    OpenAI is asked to send a final usage chunk (`stream_options.include_usage`).
"""
from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Annotated, Any

import anyio
import openai
from fastapi import APIRouter, Body, Response
from fastapi.responses import JSONResponse, StreamingResponse

from cloud.api.deps import CurrentUser, ServicesDep, api_error, charge_request
from cloud.api.services import Services
from cloud.api.tokens import AccessClaims
from core.errors import MissingAPIKeyError
from core.models.gateway import LONG_TIMEOUT

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])

#: Everything the agents send, and nothing that changes who pays or where data goes.
ALLOWED_PARAMETERS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "response_format",
        "reasoning_effort",
        "max_completion_tokens",
        "stop",
        "seed",
        "stream",
    }
)

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/llm/chat")
async def chat(
    body: Annotated[dict[str, Any], Body()], user: CurrentUser, services: ServicesDep
) -> Response:
    model = body.get("model")
    if model not in services.limits.chat_models:
        raise api_error(400, "model_not_allowed", f"The model {model!r} is not available.")
    unsupported = sorted(set(body) - ALLOWED_PARAMETERS)
    if unsupported:
        raise api_error(
            400, "unsupported_parameter", f"Unsupported parameters: {', '.join(unsupported)}."
        )
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise api_error(400, "invalid_request", "messages must be a non-empty list.")

    await charge_request(services, user)
    stream = bool(body.pop("stream", False))

    with upstream_errors():
        if not stream:
            completion = await services.gateway.chat(timeout=LONG_TIMEOUT, **body)
            await _record_usage(services, user, completion.usage)
            return JSONResponse(completion.model_dump(mode="json", exclude_unset=True))

        # Opened before the response starts, so a refused request still gets a
        # real status code rather than an error in the middle of a 200.
        chunks = await services.gateway.chat_stream(
            timeout=LONG_TIMEOUT, stream_options={"include_usage": True}, **body
        )
    return StreamingResponse(
        _relay(chunks, services, user), media_type="text/event-stream", headers=_STREAM_HEADERS
    )


@contextlib.contextmanager
def upstream_errors() -> Iterator[None]:
    """OpenAI's failures, as answers the runtime can act on."""
    try:
        yield
    except MissingAPIKeyError:
        logger.error("The platform OpenAI key is not configured")
        raise api_error(503, "llm_unavailable", "The model service is not configured.") from None
    except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
        raise api_error(400, "invalid_request", exc.message) from None
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        logger.error("OpenAI refused the platform key")
        raise api_error(503, "llm_unavailable", "The model service is unavailable.") from None
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError):
        # Already retried with backoff in core.models.llm.
        raise api_error(
            503, "llm_unavailable", "The model service is busy. Try again shortly."
        ) from None
    except openai.OpenAIError:
        logger.warning("OpenAI request failed", exc_info=True)
        raise api_error(502, "upstream_error", "The model service failed.") from None


async def _relay(
    chunks: AsyncIterable, services: Services, user: AccessClaims
) -> AsyncIterator[str]:
    usage = None
    try:
        async for chunk in chunks:
            if chunk.usage is not None:
                usage = chunk.usage
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.OpenAIError as exc:
        # Headers are long gone, so the failure travels as an event. No [DONE]
        # follows, so the client cannot mistake a cut-off answer for a whole one.
        logger.warning("A model stream failed part-way: %s", exc)
        error = {"code": "upstream_error", "message": "The model stopped responding part-way."}
        yield f"event: error\ndata: {json.dumps(error)}\n\n"
    finally:
        # Also reached when the client disconnects, and the task is being
        # cancelled. Shielded, or neither the upstream stream would be closed
        # (OpenAI would keep generating, and billing) nor its usage counted.
        with anyio.CancelScope(shield=True):
            close = getattr(chunks, "close", None)
            if close is not None:
                await close()
            await _record_usage(services, user, usage)


async def _record_usage(services: Services, user: AccessClaims, usage) -> None:
    if usage is None:
        return
    try:
        await services.usage.add_tokens(
            user.user_id,
            datetime.now(UTC).date(),
            prompt=usage.prompt_tokens or 0,
            completion=usage.completion_tokens or 0,
        )
    except Exception:  # noqa: BLE001
        # The answer was already delivered; losing its count must not fail it.
        logger.exception("Could not record token usage for user %s", user.user_id)
