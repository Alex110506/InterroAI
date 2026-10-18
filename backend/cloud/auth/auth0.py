"""
Auth0, as the API sees it: a signer of access tokens, and a source of profiles.

Signing in happens between the desktop app and Auth0, through GitHub; the API
never sees a password, a code or a refresh token. What reaches it is an access
token on every request, which it checks here against the tenant's published
keys: signed with RS256 by this tenant, meant for this API, not expired.

The token names the user only by Auth0's subject, `github|<numeric id>`. The
GitHub login and avatar come from `/userinfo`, which accepts the same token
because the app asks for the `openid profile` scopes alongside this API's
audience.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx
import jwt

_ALGORITHMS = ["RS256"]
#: Clock skew tolerated between Auth0 and this API.
_LEEWAY_SECONDS = 10
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]
_USERINFO_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class InvalidTokenError(Exception):
    """A token that is malformed, forged, expired, or meant for something else."""


class Auth0Unavailable(Exception):
    """Auth0 could not be reached, so a token could not be checked or a profile read."""


@dataclass(frozen=True)
class Auth0Config:
    #: The tenant's domain, e.g. `interroai.eu.auth0.com`.
    domain: str
    #: The desktop app's client id. It is a public client: this is not a secret.
    client_id: str
    #: This API's identifier in Auth0, which every access token must carry in `aud`.
    audience: str

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"


@dataclass(frozen=True)
class GitHubProfile:
    github_id: int
    login: str
    avatar_url: str | None


class SigningKeys(Protocol):
    """What `jwt.PyJWKClient` offers: the published key that signed a token."""

    def get_signing_key_from_jwt(self, token: str): ...


def github_id(subject: str) -> int | None:
    """The GitHub account an Auth0 subject names, or `None` if it is not a GitHub login."""
    provider, _, raw_id = subject.partition("|")
    return int(raw_id) if provider == "github" and raw_id.isdigit() else None


class Auth0:
    def __init__(
        self, config: Auth0Config, *, http: httpx.AsyncClient, jwks: SigningKeys | None = None
    ) -> None:
        self.config = config
        self._http = http
        # Caches the key set and refetches it only for a key id it has not seen,
        # which is how a key rotated at Auth0 is picked up.
        self._jwks = jwks or jwt.PyJWKClient(f"{config.issuer}.well-known/jwks.json")

    async def verify(self, token: str) -> str:
        """Check an access token and return its subject."""
        try:
            # PyJWKClient fetches over blocking HTTP when its cache misses.
            signing_key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
        except jwt.PyJWKClientConnectionError as exc:
            raise Auth0Unavailable(str(exc)) from None
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from None
        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                # A fixed list, never the token's own header: that is what stops
                # a token "signed" with HS256 and the public key from passing.
                algorithms=_ALGORITHMS,
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=_LEEWAY_SECONDS,
                options={"require": _REQUIRED_CLAIMS},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from None
        return str(claims["sub"])

    async def profile(self, token: str) -> GitHubProfile:
        """The GitHub account behind a token, read from Auth0's `/userinfo`."""
        try:
            response = await self._http.get(
                f"https://{self.config.domain}/userinfo",
                headers={"Authorization": f"Bearer {token}"},
                timeout=_USERINFO_TIMEOUT,
            )
        except httpx.TransportError as exc:
            raise Auth0Unavailable(str(exc)) from None
        if response.status_code in (401, 403):
            raise InvalidTokenError("Auth0 refused the token at /userinfo.")
        if not response.is_success:
            raise Auth0Unavailable(f"/userinfo answered {response.status_code}")
        body = response.json()
        account = github_id(str(body.get("sub", "")))
        login = body.get("nickname")
        if account is None or not login:
            raise InvalidTokenError("The token does not belong to a GitHub account.")
        return GitHubProfile(github_id=account, login=str(login), avatar_url=body.get("picture"))
