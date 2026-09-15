"""A stand-in for GitHub's OAuth endpoints, for the sign-in tests."""
from __future__ import annotations

from urllib.parse import urlencode

from cloud.api.github import GitHubIdentity


class FakeGitHub:
    def __init__(self, login: str = "Alex110506") -> None:
        self.identity = GitHubIdentity(
            id=583231, login=login, avatar_url="https://avatars.example/583231"
        )
        #: Raised by `identify` when set: GitHub refusing the code, or unreachable.
        self.failure: Exception | None = None
        self.codes: list[str] = []

    def authorize_url(self, state: str) -> str:
        return f"https://github.example/login/oauth/authorize?{urlencode({'state': state})}"

    async def identify(self, code: str) -> GitHubIdentity:
        self.codes.append(code)
        if self.failure is not None:
            raise self.failure
        return self.identity
