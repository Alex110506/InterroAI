"""
The launch token: only the app that started this runtime may use it.

The runtime listens on 127.0.0.1. That keeps other machines out, but not other
programs on this one, and not web pages open in a browser, which can send
requests to localhost too. So when the Electron app starts the runtime it makes
a random token, passes it in the environment, and sends it back with every
request: in the `X-Interroai-Token` header, or as `?token=` on a WebSocket URL,
since a browser cannot put headers on those. Anything without it is refused
before it reaches a route.

`/health` stays open. It tells the app the runtime is up, and nothing else.
"""
from __future__ import annotations

import hmac
import json
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Receive, Scope, Send

TOKEN_HEADER = "x-interroai-token"
_OPEN_PATHS = frozenset({"/health"})
#: In the application range, so a refused socket is recognisable as this refusal.
_WEBSOCKET_REFUSED = 4401
_REFUSAL = json.dumps(
    {
        "detail": {
            "code": "launch_token_required",
            "message": "This runtime only answers the InterroAI app that started it.",
        }
    }
).encode()


class LaunchTokenMiddleware:
    def __init__(self, app: ASGIApp, *, token: str) -> None:
        if not token:
            raise ValueError("A launch token must not be empty.")
        self._app = app
        self._token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self._admitted(scope):
            await self._app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # Closing before accepting refuses the handshake itself.
            await send({"type": "websocket.close", "code": _WEBSOCKET_REFUSED})
            return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_REFUSAL)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _REFUSAL})

    def _admitted(self, scope: Scope) -> bool:
        if scope["path"] in _OPEN_PATHS:
            return True
        if scope["type"] == "http" and scope["method"] == "OPTIONS":
            # A CORS preflight never carries the token; CORSMiddleware answers it.
            return True
        presented = _header(scope, TOKEN_HEADER)
        if presented is None and scope["type"] == "websocket":
            # Only for WebSockets: a URL ends up in logs and history.
            presented = _query_parameter(scope, "token")
        return presented is not None and hmac.compare_digest(presented.encode(), self._token)


def _header(scope: Scope, name: str) -> str | None:
    wanted = name.encode("latin-1")
    for key, value in scope.get("headers", []):
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


def _query_parameter(scope: Scope, name: str) -> str | None:
    values = parse_qs(scope.get("query_string", b"").decode("latin-1")).get(name)
    return values[0] if values else None
