"""
Everything a request handler needs, built once per process.

`create_app` accepts a `Services` so tests can hand in fakes. In production the
app builds one from `ApiSettings` when it starts and closes it when it stops.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta

import httpx

from cloud.api.github import GitHubOAuth
from cloud.api.signin import SignInService
from cloud.api.tokens import TokenSigner
from cloud.db.accounts import Accounts, PostgresAccounts
from cloud.db.projects import ProjectRepository
from cloud.db.session import anonymous_scope, create_engine, create_session_factory
from cloud.settings import ApiSettings

CALLBACK_PATH = "/auth/github/callback"


@dataclass
class Services:
    signin: SignInService
    signer: TokenSigner
    accounts: Accounts
    projects: ProjectRepository
    #: Sign-in cookies are marked Secure whenever the API is served over https.
    secure_cookies: bool = False
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    async def close(self) -> None:
        for close in reversed(self.closers):
            await close()


def build_services(settings: ApiSettings) -> Services:
    """Wire the production adapters. Nothing connects until the first request."""
    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)
    http = httpx.AsyncClient()
    public_url = settings.public_api_url.rstrip("/")

    signer = TokenSigner(
        settings.jwt_secret.get_secret_value(),
        issuer=public_url,
        access_ttl=timedelta(seconds=settings.access_token_ttl_seconds),
    )
    accounts = PostgresAccounts(lambda: anonymous_scope(sessions))
    github = GitHubOAuth(
        client_id=settings.github_client_id,
        client_secret=settings.github_client_secret.get_secret_value(),
        callback_url=public_url + CALLBACK_PATH,
        http=http,
    )
    signin = SignInService(
        accounts=accounts,
        github=github,
        signer=signer,
        allowlist=settings.allowlist,
        refresh_ttl=timedelta(days=settings.refresh_token_ttl_days),
    )
    return Services(
        signin=signin,
        signer=signer,
        accounts=accounts,
        projects=ProjectRepository(sessions),
        secure_cookies=public_url.startswith("https://"),
        closers=[engine.dispose, http.aclose],
    )
