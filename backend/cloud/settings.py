"""
Settings for the cloud API and the indexing worker.

Everything comes from environment variables prefixed `INTERROAI_`, or from a
`.env` file during development. In Azure the secrets among them become
Container Apps secrets backed by Key Vault references and reach the process as
the very same environment variables — so there is no Key Vault client in this
code, and nothing here changes when it is deployed.

Two classes because the two processes need different things: the worker has no
business holding the GitHub client secret or the JWT signing key.
"""
from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `backend/` is the working directory in development, so the repo-root `.env`
#: is one level up; a `.env` beside the process wins if both exist.
_ENV_FILES = ("../.env", ".env")


class WorkerSettings(BaseSettings):
    """What both cloud processes need: storage, the queue, and the platform key."""

    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=_ENV_FILES, extra="ignore"
    )

    #: The app role. Row-level security applies to it, which is the point.
    database_url: str
    #: The owner role, for migrations only. Never used by a running service.
    migrations_database_url: str | None = None

    servicebus_connection_string: SecretStr
    servicebus_queue: str = "index-jobs"

    blob_connection_string: SecretStr
    blob_container: str = "uploads"

    openai_api_key: SecretStr


class ApiSettings(WorkerSettings):
    """The worker's settings plus sign-in, tokens and quotas."""

    github_client_id: str
    github_client_secret: SecretStr
    public_api_url: str = "http://localhost:8080"

    #: HS256 key for access tokens. 32 bytes is the floor for HMAC-SHA256 to
    #: carry its full strength.
    jwt_secret: SecretStr = Field(min_length=32)
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_days: int = 30

    #: Comma-separated. Empty admits nobody: a misconfigured deployment should
    #: fail closed, not open sign-up on the platform's OpenAI key.
    allowed_github_logins: str = ""

    daily_token_quota: int = 200_000
    daily_request_quota: int = 500
    max_upload_bytes: int = 25_000_000

    @property
    def allowlist(self) -> frozenset[str]:
        """The allowed logins, lower-cased — GitHub logins are case-insensitive."""
        return frozenset(
            login.strip().lower()
            for login in self.allowed_github_logins.split(",")
            if login.strip()
        )
