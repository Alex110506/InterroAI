"""
Everything a request handler needs, built once per process.

`create_app` accepts a `Services` so tests can hand in fakes, and each test
fills in only the parts its routes use. In production the app builds the whole
thing from `ApiSettings` when it starts and closes it when it stops.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cloud.adapters.blob_uploads import BlobUploadStore
from cloud.adapters.pg_cache import PostgresEmbeddingCache
from cloud.adapters.service_bus import ServiceBusJobQueue
from cloud.api.github import GitHubOAuth
from cloud.api.job_events import JobNotifications, asyncpg_dsn
from cloud.api.signin import SignInService
from cloud.api.throttle import RateLimiter, RateLimits
from cloud.api.tokens import TokenSigner
from cloud.db.accounts import Accounts, PostgresAccounts
from cloud.db.projects import ProjectRepository
from cloud.db.session import anonymous_scope, create_engine, create_session_factory
from cloud.db.usage import PostgresUsageMeter, Quota, UsageMeter
from cloud.settings import ApiSettings
from core.index.embeddings import embed_texts
from core.index.ports import JobQueue
from core.models import llm
from core.models.gateway import ModelGateway, OpenAIGateway

CALLBACK_PATH = "/auth/github/callback"

#: How long one startup step may run before the app starts without it. The
#: Azure SDKs retry an unreachable endpoint for minutes on their own, which is
#: far longer than any platform waits for a first health check.
_STARTUP_STEP_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 25_000_000
    max_request_bytes: int = 8_000_000
    upload_url_ttl: timedelta = timedelta(minutes=15)
    sse_heartbeat_seconds: float = 15.0
    quota: Quota = Quota(requests=500, tokens=200_000)
    rates: RateLimits = RateLimits()
    chat_models: frozenset[str] = frozenset({"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"})


@dataclass
class Services:
    signin: SignInService
    signer: TokenSigner
    accounts: Accounts
    projects: ProjectRepository
    #: For the repositories that run in a user's scope, built per request.
    sessions: async_sessionmaker[AsyncSession] | None = None
    uploads: BlobUploadStore | None = None
    queue: JobQueue | None = None
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None
    usage: UsageMeter | None = None
    gateway: ModelGateway | None = None
    notifications: JobNotifications = field(default_factory=lambda: JobNotifications(None))
    limits: Limits = field(default_factory=Limits)
    limiter: RateLimiter = field(default_factory=RateLimiter)
    #: Sign-in cookies are marked Secure whenever the API is served over https.
    secure_cookies: bool = False
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    #: Run as the app starts, to get backing services ready: creating the upload
    #: container, which a fresh local stack does not have.
    startup: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    async def start(self) -> None:
        """
        Run the startup steps.

        A step that fails or runs long is logged, not fatal: with storage down,
        sign-in and everything else should still work, and the upload that
        needed storage fails with its own error. The time limit carries as much
        weight as the exception — an unreachable endpoint does not raise
        promptly, it retries in the SDK until well past any health check.
        """
        for step in self.startup:
            name = getattr(step, "__qualname__", repr(step))
            try:
                await asyncio.wait_for(step(), _STARTUP_STEP_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("A startup step timed out (%s); starting anyway", name)
            except Exception:  # noqa: BLE001
                logger.warning("A startup step failed (%s); starting anyway", name, exc_info=True)

    async def close(self) -> None:
        for close in reversed(self.closers):
            await close()


def build_services(settings: ApiSettings) -> Services:
    """Wire the production adapters. Nothing connects until the first request."""
    # The API embeds search queries and proxies chat, both on the platform key.
    llm.use_api_key(settings.openai_api_key.get_secret_value())

    engine = create_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
    sessions = create_session_factory(engine)
    http = httpx.AsyncClient()
    public_url = settings.public_api_url.rstrip("/")

    def anonymous():
        return anonymous_scope(sessions)

    signer = TokenSigner(
        settings.jwt_secret.get_secret_value(),
        issuer=public_url,
        access_ttl=timedelta(seconds=settings.access_token_ttl_seconds),
    )
    accounts = PostgresAccounts(anonymous)
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
    uploads = BlobUploadStore.from_connection_string(
        settings.blob_connection_string.get_secret_value(),
        settings.blob_container,
        max_bytes=settings.max_upload_bytes,
    )
    queue = ServiceBusJobQueue.from_connection_string(
        settings.servicebus_connection_string.get_secret_value(), settings.servicebus_queue
    )
    cache = PostgresEmbeddingCache(anonymous)
    notifications = JobNotifications(asyncpg_dsn(settings.database_url))

    async def embed_query(query: str) -> list[float]:
        [vector] = await embed_texts([query], cache=cache)
        return vector

    return Services(
        signin=signin,
        signer=signer,
        accounts=accounts,
        projects=ProjectRepository(sessions),
        sessions=sessions,
        uploads=uploads,
        queue=queue,
        embed_query=embed_query,
        usage=PostgresUsageMeter(anonymous),
        gateway=OpenAIGateway(),
        notifications=notifications,
        limits=Limits(
            max_upload_bytes=settings.max_upload_bytes,
            max_request_bytes=settings.max_request_bytes,
            upload_url_ttl=timedelta(seconds=settings.upload_url_ttl_seconds),
            sse_heartbeat_seconds=settings.sse_heartbeat_seconds,
            quota=Quota(requests=settings.daily_request_quota, tokens=settings.daily_token_quota),
            rates=RateLimits(
                sign_in=settings.sign_in_per_minute,
                chat=settings.chat_per_minute,
                search=settings.search_per_minute,
                sync=settings.sync_per_minute,
                uploads=settings.uploads_per_minute,
            ),
            chat_models=settings.chat_model_allowlist,
        ),
        secure_cookies=public_url.startswith("https://"),
        closers=[engine.dispose, http.aclose, uploads.close, queue.close, notifications.close],
        # Upload URLs point into this container, so it has to exist before the
        # first one is used. In Azure it already does, and this does nothing.
        startup=[uploads.ensure_container],
    )
