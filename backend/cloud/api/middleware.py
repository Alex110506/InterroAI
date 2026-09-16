"""
What every request to the Cloud API goes through before it reaches a route.

  * **A request id**, taken from `X-Request-ID` when a proxy set a sensible one
    and made here otherwise. It is echoed back, and attached to every log line
    written while the request runs.
  * **One access log line per request**: method, path, status, duration, and the
    user once authentication has run.
  * **A limit on request bodies.** The largest legitimate body is a sync
    manifest; anything far beyond that is refused before it is read into memory.
    Chunk uploads never come through here: they go straight to Blob Storage.

Pure ASGI rather than `BaseHTTPMiddleware`, which buffers streaming responses:
job events and chat streams have to reach the client as they happen.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cloud import observability

logger = logging.getLogger("cloud.api.access")

#: Accepted from a proxy only when it looks like an id, not text smuggled into the logs.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,128}")
_DEFAULT_MAX_BODY_BYTES = 8_000_000
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = {
            name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]
        }
        inbound = headers.get("x-request-id", "")
        request = inbound if _REQUEST_ID.fullmatch(inbound) else uuid.uuid4().hex
        request_token = observability.request_id.set(request)
        user_token = observability.user_id.set(None)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {
                    **message,
                    "headers": [*message.get("headers", []), (b"x-request-id", request.encode())],
                }
            await send(message)

        try:
            refusal = _refusal(scope, headers)
            if refusal is None:
                await self._app(scope, receive, send_with_id)
            else:
                await _refuse(send_with_id, *refusal)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            # Health probes arrive every few seconds; they are not worth a line each.
            level = logging.DEBUG if scope["path"] == "/health" else logging.INFO
            logger.log(
                level,
                "%s %s %d %.1fms",
                scope["method"],
                scope["path"],
                status,
                duration_ms,
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": duration_ms,
                    "request_id": request,
                    "user_id": observability.user_id.get(),
                },
            )
            observability.user_id.reset(user_token)
            observability.request_id.reset(request_token)


def _refusal(scope: Scope, headers: dict[str, str]) -> tuple[int, str, str] | None:
    if scope["method"] not in _BODY_METHODS:
        return None
    limit = _max_body_bytes(scope)
    declared = headers.get("content-length")
    if declared is None:
        if "transfer-encoding" in headers:
            # A body of unknown length could only be measured by reading it, and
            # every client of this API sends a length, so one is required.
            return 411, "length_required", "A request body must declare its length."
        return None
    if declared.isdigit() and int(declared) > limit:
        return 413, "request_too_large", f"The request body is over the {limit:,}-byte limit."
    return None


def _max_body_bytes(scope: Scope) -> int:
    app = scope.get("app")
    services = getattr(getattr(app, "state", None), "services", None)
    return services.limits.max_request_bytes if services is not None else _DEFAULT_MAX_BODY_BYTES


async def _refuse(send: Send, status: int, code: str, message: str) -> None:
    body = json.dumps({"detail": {"code": code, "message": message}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
