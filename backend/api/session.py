"""
The runtime's cloud session, for the Electron app.

  GET    /api/session   where the Cloud API is, who is signed in, and, while
                        nobody is, where to sign in (the Auth0 tenant)
  PUT    /api/session   finish signing in with the authorization code, PKCE
                        verifier and redirect URI the app's browser leg used
  DELETE /api/session   sign out

The app runs the browser half of sign-in, since it can open a browser and listen
on 127.0.0.1. This process redeems the code at Auth0 and keeps the tokens, so
they never pass through the app.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from core import providers
from core.errors import CloudError, CloudUnavailableError, NotSignedInError

router = APIRouter(prefix="/api/session", tags=["session"])


class Auth0Info(BaseModel):
    domain: str
    client_id: str
    audience: str


class SessionState(BaseModel):
    api_url: str
    signed_in: bool = False
    login: str | None = None
    avatar_url: str | None = None
    #: Present while signed out and the cloud is reachable: what the app needs
    #: to start signing in.
    auth0: Auth0Info | None = None


class SignInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    code_verifier: str
    redirect_uri: str


@router.get("")
async def get_session() -> SessionState:
    session = providers.cloud_session()
    state = SessionState(api_url=session.api_url)
    if not session.signed_in:
        return await _with_auth0(state)
    try:
        identity = await session.identity()
    except NotSignedInError:
        return await _with_auth0(state)
    except CloudUnavailableError:
        # Offline: still signed in as far as this machine knows.
        return state.model_copy(update={"signed_in": True})
    return state.model_copy(
        update={"signed_in": True, "login": identity.login, "avatar_url": identity.avatar_url}
    )


@router.put("")
async def sign_in(body: SignInRequest) -> SessionState:
    session = providers.cloud_session()
    try:
        identity = await session.sign_in(
            code=body.code, code_verifier=body.code_verifier, redirect_uri=body.redirect_uri
        )
    except CloudError as exc:
        status = 503 if isinstance(exc, CloudUnavailableError) else 400
        raise HTTPException(status, detail={"code": exc.code, "message": str(exc)}) from None
    return SessionState(
        api_url=session.api_url,
        signed_in=True,
        login=identity.login,
        avatar_url=identity.avatar_url,
    )


@router.delete("", status_code=204)
async def sign_out() -> Response:
    await providers.cloud_session().sign_out()
    return Response(status_code=204)


async def _with_auth0(state: SessionState) -> SessionState:
    try:
        config = await providers.cloud_session().auth_config()
    except CloudError:
        # The app shows the sign-in screen either way; starting sign-in will say why it can't.
        return state
    return state.model_copy(
        update={
            "auth0": Auth0Info(
                domain=config.domain, client_id=config.client_id, audience=config.audience
            )
        }
    )
