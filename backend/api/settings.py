"""Settings API — the user's name, stored in ~/.interroai/config.json."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import app_config

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsSaveRequest(BaseModel):
    name: str = Field(default="", max_length=100)


class SettingsResponse(BaseModel):
    name: str


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Return the stored user name. Model access is the platform's, not this machine's."""
    return SettingsResponse(name=app_config.get("user_name", ""))


@router.post("")
async def save_settings(body: SettingsSaveRequest) -> dict:
    """Persist the user name to ~/.interroai/config.json."""
    app_config.set("user_name", body.name)
    return {"ok": True}
