"""
A stand-in for an Auth0 tenant, for both halves of sign-in.

It signs real RS256 tokens with a key of its own, so the Cloud API's checks run
for real against it, and it answers the three endpoints the code calls over
HTTP: `/userinfo` (the API, for a GitHub profile), `/oauth/token` and
`/oauth/revoke` (the runtime, to sign in, refresh and sign out). Refresh tokens
rotate with reuse detection, as Auth0's do when rotation is switched on.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from cloud.auth.auth0 import Auth0, Auth0Config, github_id
from cloud.auth.identity import IdentityService

DOMAIN = "interroai.auth0.example"
AUDIENCE = "https://api.interroai.example"
CLIENT_ID = "desktop-app"


@cache
def _signing_key() -> rsa.RSAPrivateKey:
    # Generating an RSA key takes a noticeable fraction of a second; one per run is enough.
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@dataclass
class _Grant:
    github_id: int
    code_verifier: str
    redirect_uri: str


class FakeJWKS:
    """Answers like `jwt.PyJWKClient`: the public key for any token it signed."""

    def __init__(self) -> None:
        #: Raised instead of answering when set: Auth0's key set unreachable, say.
        self.failure: Exception | None = None

    def get_signing_key_from_jwt(self, token: str) -> SimpleNamespace:
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(key=_signing_key().public_key())


class FakeAuth0:
    def __init__(self) -> None:
        self.config = Auth0Config(domain=DOMAIN, client_id=CLIENT_ID, audience=AUDIENCE)
        self.jwks = FakeJWKS()
        #: github_id → (login, avatar_url), as `/userinfo` reports them.
        self.profiles: dict[int, tuple[str, str | None]] = {}
        self.userinfo_calls = 0
        #: When set, every endpoint answers 503.
        self.down = False
        self.revoked: list[str] = []
        self._codes: dict[str, _Grant] = {}
        self._live_refresh: dict[str, int] = {}
        self._spent_refresh: set[str] = set()
        self.transport = httpx.MockTransport(self._handle)

    # ── Arranging ────────────────────────────────────────────────────────────

    def add_user(
        self, github_id: int, login: str, avatar_url: str | None = None
    ) -> FakeAuth0:
        self.profiles[github_id] = (login, avatar_url)
        return self

    def issue_code(self, github_id: int, *, code_verifier: str, redirect_uri: str) -> str:
        """What the browser leg would bring back to the app's loopback listener."""
        code = secrets.token_urlsafe(16)
        self._codes[code] = _Grant(github_id, code_verifier, redirect_uri)
        return code

    def token(
        self,
        github_id: int | None = None,
        *,
        subject: str | None = None,
        audience: str | list[str] | None = None,
        issuer: str | None = None,
        expires_in: int = 900,
        algorithm: str = "RS256",
        key: object | None = None,
    ) -> str:
        """An access token as Auth0 would sign it, or deliberately wrong in one way."""
        now = int(time.time())
        claims = {
            "sub": subject if subject is not None else f"github|{github_id}",
            "aud": audience if audience is not None else [AUDIENCE, f"https://{DOMAIN}/userinfo"],
            "iss": issuer if issuer is not None else self.config.issuer,
            "iat": now,
            "exp": now + expires_in,
            "scope": "openid profile offline_access",
        }
        return jwt.encode(claims, key if key is not None else _signing_key(), algorithm=algorithm)

    def bearer(self, github_id: int) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(github_id)}"}

    # ── Wiring ───────────────────────────────────────────────────────────────

    def client(self) -> Auth0:
        """The real `Auth0` client, pointed at this fake."""
        return Auth0(self.config, http=httpx.AsyncClient(transport=self.transport), jwks=self.jwks)

    def identity(self, accounts, allowlist: tuple[str, ...] = ("alex110506",)) -> IdentityService:
        return IdentityService(
            auth0=self.client(), accounts=accounts, allowlist=frozenset(allowlist)
        )

    # ── The endpoints ────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            return httpx.Response(503, json={"error": "temporarily_unavailable"})
        if request.url.host != DOMAIN:
            return httpx.Response(404)
        if request.method == "GET" and request.url.path == "/userinfo":
            return self._userinfo(request)
        if request.method == "POST" and request.url.path == "/oauth/token":
            return self._token_endpoint(_form(request))
        if request.method == "POST" and request.url.path == "/oauth/revoke":
            self.revoked.append(_form(request).get("token", ""))
            self._live_refresh.pop(_form(request).get("token", ""), None)
            return httpx.Response(200)
        return httpx.Response(404)

    def _userinfo(self, request: httpx.Request) -> httpx.Response:
        self.userinfo_calls += 1
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        try:
            subject = jwt.decode(token, options={"verify_signature": False})["sub"]
        except jwt.PyJWTError:
            return httpx.Response(401)
        account = github_id(subject)
        profile = self.profiles.get(account) if account is not None else None
        if profile is None:
            return httpx.Response(401)
        login, avatar_url = profile
        return httpx.Response(200, json={"sub": subject, "nickname": login, "picture": avatar_url})

    def _token_endpoint(self, form: dict[str, str]) -> httpx.Response:
        if form.get("client_id") != CLIENT_ID:
            return _refusal(401, "invalid_client")
        if form.get("grant_type") == "authorization_code":
            grant = self._codes.pop(form.get("code", ""), None)
            if (
                grant is None
                or grant.code_verifier != form.get("code_verifier")
                or grant.redirect_uri != form.get("redirect_uri")
            ):
                return _refusal(403, "invalid_grant")
            return self._tokens_for(grant.github_id)
        if form.get("grant_type") == "refresh_token":
            presented = form.get("refresh_token", "")
            if presented in self._spent_refresh:
                # Reuse: the whole family goes, as with Auth0's reuse detection.
                self._live_refresh.clear()
                return _refusal(403, "invalid_grant")
            github_id = self._live_refresh.pop(presented, None)
            if github_id is None:
                return _refusal(403, "invalid_grant")
            self._spent_refresh.add(presented)
            return self._tokens_for(github_id)
        return _refusal(400, "unsupported_grant_type")

    def _tokens_for(self, github_id: int) -> httpx.Response:
        refresh_token = secrets.token_urlsafe(24)
        self._live_refresh[refresh_token] = github_id
        return httpx.Response(
            200,
            json={
                "access_token": self.token(github_id),
                "refresh_token": refresh_token,
                "expires_in": 900,
                "token_type": "Bearer",
            },
        )


def _form(request: httpx.Request) -> dict[str, str]:
    content_type = request.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        return json.loads(request.content or b"{}")
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


def _refusal(status: int, error: str) -> httpx.Response:
    return httpx.Response(status, json={"error": error, "error_description": error})
