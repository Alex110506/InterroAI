"""
Settings for the cloud API and the indexing worker.

Everything comes from environment variables prefixed `INTERROAI_`, or from a
`.env` file during development. In Azure the secrets among them become
Container Apps secrets backed by Key Vault references and reach the process as
the very same environment variables — so there is no Key Vault client in this
code, and nothing here changes when it is deployed.

Two classes because the two processes need different things: the worker checks
no access tokens, so it has no business knowing the Auth0 tenant or the allowlist.
"""
from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `backend/` is the working directory in development, so the repo-root `.env`
#: is one level up; a `.env` beside the process wins if both exist.
_ENV_FILES = ("../.env", ".env")


class MigrationSettings(BaseSettings):
    """Only what Alembic needs: the owner role's connection string."""

    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=_ENV_FILES, extra="ignore"
    )

    migrations_database_url: str


class WorkerSettings(BaseSettings):
    """What both cloud processes need: storage, the queue, and the platform key."""

    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=_ENV_FILES, extra="ignore"
    )

    #: The app role. Row-level security applies to it, which is the point.
    database_url: str
    #: The owner role, for migrations only. Never used by a running service.
    migrations_database_url: str | None = None

    #: Connections this process holds, and how many more it may open in a burst.
    #: Every replica of the API and the worker draws on one allowance — a
    #: Postgres B1ms permits 35 user connections — so each keeps only what it
    #: needs. The worker handles one job at a time, so it needs very few.
    database_pool_size: int = 2
    database_max_overflow: int = 2

    servicebus_connection_string: SecretStr
    servicebus_queue: str = "index-jobs"
    #: Must match the queue's MaxDeliveryCount (infra/local/servicebus/config.json
    #: locally; Terraform in Azure), or the worker gives up too early or never.
    max_delivery_count: int = 5

    blob_connection_string: SecretStr
    blob_container: str = "uploads"
    #: The largest upload accepted. Checked when a job is created and again by
    #: the worker before it reads one: a SAS URL cannot cap a blob's size.
    max_upload_bytes: int = 25_000_000

    openai_api_key: SecretStr

    #: "json" in the images, one object per line for Log Analytics; "text" reads
    #: better in a terminal.
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"


class ApiSettings(WorkerSettings):
    """The worker's settings plus Auth0, the allowlist and quotas."""

    #: The Auth0 tenant the desktop app signs in with, e.g. `interroai.eu.auth0.com`.
    #: Access tokens are checked against the keys it publishes.
    auth0_domain: str
    #: This API's identifier in Auth0: the audience every access token must carry.
    auth0_audience: str
    #: The desktop app's Auth0 client id. A public client, so not a secret; the
    #: app reads it from `/auth/config` rather than having it built in.
    auth0_client_id: str

    #: The API serves many requests at once, so it holds more than the worker.
    database_pool_size: int = 5
    database_max_overflow: int = 5

    #: Comma-separated. Empty admits nobody: a misconfigured deployment should
    #: fail closed, not open sign-up on the platform's OpenAI key.
    allowed_github_logins: str = ""

    daily_token_quota: int = 200_000
    daily_request_quota: int = 500

    #: Comma-separated OpenAI model ids `/llm/chat` accepts: the ones the agents use.
    chat_models: str = "gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna"
    #: How long an upload URL stays usable.
    upload_url_ttl_seconds: int = 15 * 60
    #: Idle time after which a job's event stream sends a keep-alive comment.
    sse_heartbeat_seconds: float = 15.0

    #: The largest request body accepted. The biggest legitimate one is a sync
    #: manifest, a path and a hash per file; chunks go to Blob Storage instead.
    max_request_bytes: int = 8_000_000

    #: Per-minute rate limits per user (cloud/api/throttle.py).
    chat_per_minute: int = 60
    search_per_minute: int = 120
    sync_per_minute: int = 30
    uploads_per_minute: int = 30

    @property
    def chat_model_allowlist(self) -> frozenset[str]:
        return frozenset(model.strip() for model in self.chat_models.split(",") if model.strip())

    @property
    def allowlist(self) -> frozenset[str]:
        """The allowed logins, lower-cased — GitHub logins are case-insensitive."""
        return frozenset(
            login.strip().lower()
            for login in self.allowed_github_logins.split(",")
            if login.strip()
        )
