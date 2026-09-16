"""Settings API — stores user name (config file) and API key (OS keychain)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import app_config
from core.local.security import retrieve_openai_key, store_openai_key
from core.settings import get_runtime_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsSaveRequest(BaseModel):
    name: str = Field(default="", max_length=100)
    # Accept both camelCase (from JS) and snake_case
    apiKey: str = Field(default="", alias="apiKey")

    model_config = {"populate_by_name": True}


class SettingsResponse(BaseModel):
    name: str
    has_api_key: bool
    #: "cloud" means the platform holds the model key, so the app hides the key field.
    mode: str


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Return current user name, whether an API key is stored, and the runtime's mode."""
    return SettingsResponse(
        name=app_config.get("user_name", ""),
        has_api_key=bool(retrieve_openai_key()),
        mode=get_runtime_settings().mode,
    )


@router.post("")
async def save_settings(body: SettingsSaveRequest) -> dict:
    """
    Persist user name to ~/.interroai/config.json.
    Persist API key to the OS keychain via keyring.
    The key is never echoed back to the client.
    """
    app_config.set("user_name", body.name)

    if body.apiKey:
        store_openai_key(body.apiKey)

    return {"ok": True}
