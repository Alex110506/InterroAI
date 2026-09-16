"""
The runtime's signed-in session with the Cloud API.

It holds two credentials. The access token lives only in memory and lasts
fifteen minutes. The refresh token lasts thirty days and lives in the OS
keychain, never in a file, so the app stays signed in across restarts.

Every call goes through `request` or `stream`, which attach the access token
and, when the API answers 401, refresh once and retry. Many requests can meet an
expired token at the same moment, and they share a single refresh. Refresh
tokens rotate, so a second refresh with the same token would look to the API
like a stolen token being replayed, and would end every session this user has.

Signing in begins in the Electron app, which runs the browser half (PKCE and the
loopback listener) and hands the login code and its verifier to this process
through `PUT /api/session`. The tokens themselves never pass through the app.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from core.errors import CloudError, CloudUnavailableError, NotSignedInError, QuotaExceededError
from core.local import security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
#: Refresh this long before the access token expires, so a request already on
#: its way does not arrive carrying a token that has just lapsed.
_EXPIRY_MARGIN_SECONDS = 30.0


@dataclass(frozen=True)
class Identity:
    id: str
    login: str
    avatar_url: str | None


class CloudSession:
    def __init__(self, api_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        #: Tests route requests to a fake or to the cloud app itself.
        self._transport = transport
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._identity: Identity | None = None
        self._refresh_lock = asyncio.Lock()

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def signed_in(self) -> bool:
        """Whether this machine holds a session. The API may since have ended it."""
        return (
            self._access_token is not None
            or security.retrieve_refresh_token(self.api_url) is not None
        )

    async def identity(self, *, refresh: bool = False) -> Identity:
        if self._identity is None or refresh:
            body = (await self.request("GET", "/me")).json()
            self._identity = Identity(
                id=str(body["id"]), login=str(body["login"]), avatar_url=body.get("avatar_url")
            )
        return self._identity

    # ── Signing in and out ───────────────────────────────────────────────────

    async def sign_in(self, *, code: str, code_verifier: str) -> Identity:
        """Redeem the login code the app's browser leg received."""
        response = await self._send(
            "POST",
            "/auth/token",
            json={"grant_type": "authorization_code", "code": code, "code_verifier": code_verifier},
        )
        if not response.is_success:
            raise cloud_error(response)
        self._adopt(response.json())
        return await self.identity(refresh=True)

    def adopt_tokens(self, *, access_token: str, refresh_token: str, expires_in: float) -> None:
        """Take over a token pair: from signing in, from a refresh, or from a test."""
        security.store_refresh_token(self.api_url, refresh_token)
        self._access_token = access_token
        self._expires_at = time.monotonic() + expires_in

    async def sign_out(self) -> None:
        refresh_token = security.retrieve_refresh_token(self.api_url)
        self._forget()
        if not refresh_token:
            return
        try:
            await self._send("POST", "/auth/logout", json={"refresh_token": refresh_token})
        except CloudUnavailableError:
            # Signed out here all the same; the token expires on the server in time.
            logger.info("Signed out locally; the Cloud API was unreachable to revoke the session")

    # ── Requests ─────────────────────────────────────────────────────────────

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """An authorised request. Anything but a 2xx answer raises a `CloudError`."""
        async with self._client() as client:
            return await self._authorised(client, method, path, stream=False, **kwargs)

    @contextlib.asynccontextmanager
    async def stream(self, method: str, path: str, **kwargs: Any) -> AsyncIterator[httpx.Response]:
        """An authorised streaming request, yielded once its status is known to be 2xx."""
        async with self._client() as client:
            response = await self._authorised(client, method, path, stream=True, **kwargs)
            try:
                yield response
            finally:
                await response.aclose()

    async def _authorised(
        self, client: httpx.AsyncClient, method: str, path: str, *, stream: bool, **kwargs: Any
    ) -> httpx.Response:
        token = await self._access()
        response = await self._attempt(client, method, path, token, stream=stream, **kwargs)
        if response.status_code == 401:
            await self._refresh(stale=token)
            token = await self._access()
            response = await self._attempt(client, method, path, token, stream=stream, **kwargs)
        if response.is_success:
            return response
        raise cloud_error(response)

    async def _attempt(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        token: str,
        *,
        stream: bool,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        request = client.build_request(
            method, path, headers={**(headers or {}), "Authorization": f"Bearer {token}"}, **kwargs
        )
        response = await _dispatch(client, request, stream=stream)
        if stream and not response.is_success:
            # The error body is small, and needed to say what went wrong.
            await response.aread()
            await response.aclose()
        return response

    async def _access(self) -> str:
        if not self._fresh():
            await self._refresh(stale=self._access_token)
        return self._access_token

    async def _refresh(self, *, stale: str | None) -> None:
        async with self._refresh_lock:
            if self._access_token != stale and self._fresh():
                return  # another request refreshed while this one waited
            refresh_token = security.retrieve_refresh_token(self.api_url)
            if not refresh_token:
                self._forget()
                raise NotSignedInError()

            response = await self._send(
                "POST",
                "/auth/token",
                json={"grant_type": "refresh_token", "refresh_token": refresh_token},
            )
            if response.status_code in (400, 401, 403):
                # Spent, expired, revoked or refused: over on the server, so over here.
                self._forget()
                message = error_detail(response).get("message")
                raise NotSignedInError(message if response.status_code == 403 else None)
            if not response.is_success:
                raise cloud_error(response)
            self._adopt(response.json())

    def _fresh(self) -> bool:
        return (
            self._access_token is not None
            and time.monotonic() < self._expires_at - _EXPIRY_MARGIN_SECONDS
        )

    def _adopt(self, body: dict) -> None:
        self.adopt_tokens(
            access_token=body["access_token"],
            refresh_token=body["refresh_token"],
            expires_in=float(body["expires_in"]),
        )

    def _forget(self) -> None:
        self._access_token = None
        self._expires_at = 0.0
        self._identity = None
        security.delete_refresh_token(self.api_url)

    def _client(self) -> httpx.AsyncClient:
        # One client per call rather than one per session: a client's connection
        # pool belongs to the event loop that opened it, and the session outlives
        # any single loop in tests.
        return httpx.AsyncClient(
            base_url=self.api_url, timeout=_DEFAULT_TIMEOUT, transport=self._transport
        )

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """An unauthenticated request, for the sign-in endpoints themselves."""
        async with self._client() as client:
            request = client.build_request(method, path, **kwargs)
            return await _dispatch(client, request, stream=False)


async def _dispatch(
    client: httpx.AsyncClient, request: httpx.Request, *, stream: bool
) -> httpx.Response:
    try:
        return await client.send(request, stream=stream)
    except httpx.TransportError as exc:
        raise CloudUnavailableError() from exc


def error_detail(response: httpx.Response) -> dict:
    """The `detail` object of a Cloud API error, or {} for any other body."""
    try:
        body = response.json()
    except ValueError:
        return {}
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, dict) else {}


def cloud_error(response: httpx.Response) -> CloudError:
    """A Cloud API refusal, as the error the runtime raises for it."""
    detail = error_detail(response)
    message, code = detail.get("message"), detail.get("code")
    if response.status_code == 401:
        return NotSignedInError()
    if response.status_code == 429:
        if code == "rate_limited":
            # A burst, not the day's allowance: worth retrying in a moment.
            return CloudError(message, code=code, details=detail)
        return QuotaExceededError(message, details=detail)
    if response.status_code >= 500:
        return CloudUnavailableError(message, code=code, details=detail)
    return CloudError(
        message or f"The InterroAI cloud refused the request ({response.status_code}).",
        code=code,
        details=detail,
    )
