"""
The API's own credentials: access tokens, sign-in state, one-time secrets, PKCE.

Three kinds of credential leave this API, and none can stand in for another:

  * **Access tokens** — JWTs (HS256), fifteen minutes. The API checks them
    without reading the database, which is why they are short-lived: the only
    way to revoke one is to wait for it to expire.
  * **Sign-in state** — also a JWT, with a different audience, carried through
    GitHub's `state` parameter. It tells the callback where the app is
    listening and which PKCE challenge to bind the login code to, so no table
    holds half-finished sign-ins.
  * **Login codes and refresh tokens** — opaque random strings. Only their
    sha256 is stored, so a leaked database backup signs nobody in.
"""
from __future__ import annotations

import base64
import hashlib
import re
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from urllib.parse import urlsplit

import jwt

ACCESS_AUDIENCE = "interroai-api"
SIGNIN_STATE_AUDIENCE = "interroai-signin-state"
_ALGORITHM = "HS256"
#: Clock skew tolerated between API replicas.
_LEEWAY_SECONDS = 10
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud"]

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class InvalidTokenError(Exception):
    """A token that is malformed, forged, expired, or meant for something else."""


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    login: str


@dataclass(frozen=True)
class SignInState:
    #: Where the app's loopback listener waits for the login code.
    redirect_uri: str
    #: The app's PKCE challenge. The login code is bound to it.
    code_challenge: str
    #: The app's own `state`, handed back untouched so the app can match the reply.
    app_state: str
    #: Also set as a cookie in the browser that started the sign-in. The
    #: callback requires both, so a sign-in cannot be finished in another browser.
    nonce: str


_SIGNIN_STATE_FIELDS = tuple(field.name for field in fields(SignInState))


class TokenSigner:
    def __init__(
        self,
        secret: str,
        *,
        issuer: str,
        access_ttl: timedelta,
        clock: Clock = utc_now,
    ) -> None:
        self._secret = secret
        self._issuer = issuer
        self._access_ttl = access_ttl
        self._clock = clock

    @property
    def access_ttl_seconds(self) -> int:
        return int(self._access_ttl.total_seconds())

    def access_token(self, user_id: str, login: str) -> str:
        return self._encode({"sub": user_id, "login": login}, ACCESS_AUDIENCE, self._access_ttl)

    def read_access_token(self, token: str) -> AccessClaims:
        claims = self._decode(token, ACCESS_AUDIENCE)
        try:
            return AccessClaims(user_id=str(claims["sub"]), login=str(claims["login"]))
        except KeyError:
            raise InvalidTokenError("The token names no user.") from None

    def signin_state(self, state: SignInState, ttl: timedelta) -> str:
        return self._encode(asdict(state), SIGNIN_STATE_AUDIENCE, ttl)

    def read_signin_state(self, token: str) -> SignInState:
        claims = self._decode(token, SIGNIN_STATE_AUDIENCE)
        try:
            return SignInState(**{name: str(claims[name]) for name in _SIGNIN_STATE_FIELDS})
        except KeyError:
            raise InvalidTokenError("The sign-in state is incomplete.") from None

    def _encode(self, claims: dict, audience: str, ttl: timedelta) -> str:
        now = self._clock()
        payload = {**claims, "iss": self._issuer, "aud": audience, "iat": now, "exp": now + ttl}
        return jwt.encode(payload, self._secret, algorithm=_ALGORITHM)

    def _decode(self, token: str, audience: str) -> dict:
        try:
            return jwt.decode(
                token,
                self._secret,
                # A fixed list, never the token's own header: that is what
                # stops a token signed with "none" from being accepted.
                algorithms=[_ALGORITHM],
                audience=audience,
                issuer=self._issuer,
                leeway=_LEEWAY_SECONDS,
                options={"require": _REQUIRED_CLAIMS},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from None


# ── One-time secrets ─────────────────────────────────────────────────────────


def new_secret() -> str:
    """256 random bits, URL-safe: a login code, a refresh token, a nonce."""
    return secrets.token_urlsafe(32)


def digest(secret: str) -> str:
    """What the database holds in place of a secret."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


# ── PKCE (RFC 7636) ──────────────────────────────────────────────────────────

#: §4.1: 43–128 characters from the unreserved set.
_VERIFIER = re.compile(r"[A-Za-z0-9\-._~]{43,128}")
#: An S256 challenge is base64url(sha256(verifier)) without padding: 43 characters.
_S256_CHALLENGE = re.compile(r"[A-Za-z0-9\-_]{43}")


def s256_challenge(verifier: str) -> str:
    hashed = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(hashed).rstrip(b"=").decode("ascii")


def is_valid_challenge(challenge: str) -> bool:
    return _S256_CHALLENGE.fullmatch(challenge) is not None


def verifier_matches(verifier: str, challenge: str) -> bool:
    if _VERIFIER.fullmatch(verifier) is None:
        return False
    return secrets.compare_digest(s256_challenge(verifier), challenge)


# ── Loopback redirects (RFC 8252 §7.3) ───────────────────────────────────────


def is_loopback_redirect(uri: str) -> bool:
    """
    Whether *uri* is an address the desktop app can be listening on.

    The login code is sent there, so this check is what stops a crafted sign-in
    link from delivering a code to someone else's server: only a process on the
    user's own machine can receive it, and only the app holds the PKCE verifier
    that redeems it. IP literals only — `localhost` is a name, and a name can be
    pointed somewhere else (RFC 8252 §8.3).
    """
    try:
        parts = urlsplit(uri)
        port = parts.port
        host = ip_address(parts.hostname or "")
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and host.is_loopback
        and bool(port)
        and not (parts.username or parts.password or parts.query or parts.fragment)
    )
