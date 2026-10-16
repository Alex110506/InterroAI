"""
GitHub, for identity only.

The API asks GitHub who the user is and nothing else. No scopes are requested,
so the token GitHub issues reads public profile data and no repository, and it
is dropped as soon as the profile has been read. The user's code never passes
through here: in this architecture edits land on their own disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_URL = "https://api.github.com/user"
_TIMEOUT = httpx.Timeout(10.0)


class GitHubError(Exception):
    """GitHub refused the code, or could not be reached."""


@dataclass(frozen=True)
class GitHubIdentity:
    #: The identity. Logins can be renamed; this id cannot.
    id: int
    login: str
    avatar_url: str | None


class GitHubOAuth:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        callback_url: str,
        http: httpx.AsyncClient,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._callback_url = callback_url
        self._http = http

    def authorize_url(self, state: str) -> str:
        query = urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self._callback_url,
                "state": state,
                # Access is by allowlist; GitHub's sign-up page is no use to anyone here.
                "allow_signup": "false",
            }
        )
        return f"{AUTHORIZE_URL}?{query}"

    async def identify(self, code: str) -> GitHubIdentity:
        """Exchange GitHub's code for a token, and the token for the user's profile."""
        try:
            exchanged = await self._http.post(
                TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._callback_url,
                },
                headers={"Accept": "application/json"},
                timeout=_TIMEOUT,
            )
            exchanged.raise_for_status()
            body = exchanged.json()
            token = body.get("access_token")
            if not token:
                # GitHub answers a bad or reused code with 200 and an `error` field.
                raise GitHubError(body.get("error", "GitHub issued no access token"))

            profile = await self._http.get(
                USER_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=_TIMEOUT,
            )
            profile.raise_for_status()
            user = profile.json()
            return GitHubIdentity(
                id=int(user["id"]), login=str(user["login"]), avatar_url=user.get("avatar_url")
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise GitHubError(f"GitHub sign-in failed: {exc}") from exc
