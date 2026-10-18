"""
Who a request runs as: from an Auth0 access token to a user row and its claims.

Users are created on first sight. Auth0 names an account only by its subject,
so the first request a GitHub account makes reads its login and avatar from
`/userinfo`, checks the allowlist, and inserts the row; every later request
finds the row by GitHub id and asks Auth0 nothing. Nobody off the allowlist
gets a row, and an account that was just refused is remembered for a few
minutes, so a stranger's token cannot make this API call Auth0 on every request.

The allowlist is checked on every request, against the stored login. Taking
someone off it ends their access with their next request, not their next
sign-in.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from cloud.auth.accounts import Accounts, UserRecord
from cloud.auth.auth0 import Auth0Unavailable, GitHubProfile, github_id

#: How long a refused account is answered from memory instead of from Auth0.
_REFUSAL_MEMORY_SECONDS = 300.0

_NOT_ALLOWED = "This GitHub account is not allowed to use InterroAI."


class TokenAuthority(Protocol):
    async def verify(self, token: str) -> str: ...

    async def profile(self, token: str) -> GitHubProfile: ...


class AccessDeniedError(Exception):
    """A valid token for an account that may not use InterroAI."""


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    login: str


class IdentityService:
    def __init__(
        self,
        *,
        auth0: TokenAuthority,
        accounts: Accounts,
        allowlist: frozenset[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._auth0 = auth0
        self._accounts = accounts
        #: Lower-cased: GitHub logins are case-insensitive.
        self._allowlist = frozenset(login.lower() for login in allowlist)
        self._clock = clock
        self._refused: dict[str, float] = {}

    async def resolve(self, token: str) -> AccessClaims:
        subject = await self._auth0.verify(token)
        account = github_id(subject)
        if account is None:
            raise AccessDeniedError("InterroAI signs in with GitHub only.")

        user = await self._accounts.get_by_github_id(account)
        if user is None:
            if self._recently_refused(subject):
                raise AccessDeniedError(_NOT_ALLOWED)
            profile = await self._auth0.profile(token)
            if not self._allowed(profile.login):
                # No user row is created for someone who is refused.
                self._refused[subject] = self._clock() + _REFUSAL_MEMORY_SECONDS
                raise AccessDeniedError(_NOT_ALLOWED)
            user = await self._accounts.upsert_github_user(
                account, profile.login, profile.avatar_url
            )
        if not self._allowed(user.login):
            raise AccessDeniedError(_NOT_ALLOWED)
        return AccessClaims(user_id=user.id, login=user.login)

    async def refresh_profile(self, token: str, claims: AccessClaims) -> UserRecord:
        """
        Re-read the login and avatar from Auth0, so a GitHub rename is followed.

        The app does this once per start (`GET /me`). With Auth0 unreachable the
        stored profile is answered instead: a stale avatar is no reason to fail.
        """
        try:
            profile = await self._auth0.profile(token)
        except Auth0Unavailable:
            user = await self._accounts.get_user(claims.user_id)
            if user is None:
                raise
            return user
        if not self._allowed(profile.login):
            raise AccessDeniedError(_NOT_ALLOWED)
        return await self._accounts.upsert_github_user(
            profile.github_id, profile.login, profile.avatar_url
        )

    def _allowed(self, login: str) -> bool:
        return login.lower() in self._allowlist

    def _recently_refused(self, subject: str) -> bool:
        until = self._refused.get(subject)
        if until is None:
            return False
        if self._clock() < until:
            return True
        del self._refused[subject]
        return False
