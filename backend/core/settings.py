"""
Runtime settings: which build this process is — fully local, or a cloud client.

Read from environment variables prefixed `INTERROAI_` and from the repo-root
`.env` that docker compose also reads. `core/providers.py` is the consumer that
matters: it turns `mode` into concrete implementations of the ports.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTERROAI_", env_file=("../.env", ".env"), extra="ignore"
    )

    #: local: OpenAI directly and Chroma on disk. cloud: through the Cloud API.
    mode: Literal["local", "cloud"] = "local"
    #: The Cloud API, used only when `mode == "cloud"`.
    api_url: str = "http://localhost:8080"


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    """The process's settings, read once."""
    return RuntimeSettings()
