"""
Runtime settings: which build this process is (fully local, or a cloud client)
and, when the Electron app started it, how it keeps everything else out.

Read from environment variables prefixed `INTERROAI_` and from the repo-root
`.env` that docker compose also reads. `core/providers.py` turns `mode` into
concrete implementations of the ports; `main.py` turns the rest into middleware.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=("../.env", ".env"), extra="ignore"
    )

    #: local: OpenAI directly and Chroma on disk. cloud: through the Cloud API.
    mode: Literal["local", "cloud"] = "local"
    #: The Cloud API, used only when `mode == "cloud"`.
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
