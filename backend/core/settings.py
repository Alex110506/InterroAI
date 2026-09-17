"""
Runtime settings: where the Cloud API is and, when the Electron app started this
process, how it keeps everything else out.

Read from environment variables prefixed `INTERROAI_` and from the repo-root
`.env` that docker compose also reads. `core/providers.py` turns `api_url` into
the cloud clients; `main.py` turns the rest into middleware.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=("../.env", ".env"), extra="ignore"
    )

    #: The Cloud API this runtime uses for models and for the index.
    api_url: str = "http://localhost:8080"

    #: Made by the Electron app each time it starts this process, and required
    #: on every request after that (`api/guard.py`). None when the runtime is
    #: started by hand for development.
    launch_token: SecretStr | None = None
    #: Comma-separated origins a browser context may call this runtime from.
    #: The app sets its own; "*" when started by hand.
    allowed_origins: str = "*"

    @field_validator("launch_token", mode="before")
    @classmethod
    def _empty_means_none(cls, value):
        return value or None

    @property
    def origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    """The process's settings, read once."""
    return RuntimeSettings()
