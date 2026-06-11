"""Settings API — stores user name (config file) and API key (OS keychain)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import app_config
from core.security import store_openai_key, retrieve_openai_key

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsSaveRequest(BaseModel):
    name: str = Field(default="", max_length=100)
    # Accept both camelCase (from JS) and snake_case
    apiKey: str = Field(default="", alias="apiKey")

    model_config = {"populate_by_name": True}


class SettingsResponse(BaseModel):
    name: str
    has_api_key: bool


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Return current user name and whether an API key is stored."""
    return SettingsResponse(
        name=app_config.get("user_name", ""),
        has_api_key=bool(retrieve_openai_key()),
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
