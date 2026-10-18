"""
`/auth/config` and `/me`.

Signing in happens between the desktop app and Auth0, so the API serves no
sign-in pages and issues no tokens. It tells the app where to sign in, and,
once it has, who it is.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from cloud.api.deps import BearerToken, CurrentUser, ServicesDep, api_error
from cloud.auth.identity import AccessDeniedError

router = APIRouter(tags=["auth"])


class AuthConfigResponse(BaseModel):
    domain: str
    client_id: str
    audience: str


class MeResponse(BaseModel):
    id: str
    login: str
    avatar_url: str | None


@router.get("/auth/config")
async def auth_config(services: ServicesDep) -> AuthConfigResponse:
    """Where the app signs in. Public: none of it is a secret, and nobody is signed in yet."""
    config = services.auth0
    if config is None:
        raise api_error(503, "auth_unavailable", "Sign-in is not configured.")
    return AuthConfigResponse(
        domain=config.domain, client_id=config.client_id, audience=config.audience
    )


@router.get("/me")
async def me(user: CurrentUser, token: BearerToken, services: ServicesDep) -> MeResponse:
    # The app asks once per start, so this is where a GitHub rename is picked up.
    try:
        record = await services.identity.refresh_profile(token, user)
    except AccessDeniedError as exc:
        raise api_error(403, "access_denied", str(exc)) from None
    return MeResponse(id=record.id, login=record.login, avatar_url=record.avatar_url)
