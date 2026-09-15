"""
Sign-in routes, and `/me`.

The flow and the reasoning behind it live in `cloud/api/signin.py`. These
handlers only translate between HTTP and it. The two browser steps answer a
person with a page; the token endpoints answer the app with JSON.
"""
from __future__ import annotations

import html
from typing import Annotated, Literal

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict

from cloud.api.deps import CurrentUser, ServicesDep, unauthorized
from cloud.api.services import CALLBACK_PATH
from cloud.api.signin import SIGNIN_TTL, SignInError

router = APIRouter(tags=["auth"])

_NONCE_COOKIE = "interroai_signin"
_COOKIE_PATH = "/auth/github"


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_type: Literal["authorization_code", "refresh_token"]
    code: str | None = None
    code_verifier: str | None = None
    refresh_token: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: int
    refresh_token: str


class SignOutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class MeResponse(BaseModel):
    id: str
    login: str
    avatar_url: str | None


@router.get("/auth/github/start")
async def start_sign_in(
    services: ServicesDep,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    code_challenge_method: str = "S256",
) -> Response:
    try:
        started = services.signin.start(
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            app_state=state,
        )
    except SignInError as exc:
        return _browser_error(exc)

    response = RedirectResponse(started.github_url, status_code=302)
    response.set_cookie(
        _NONCE_COOKIE,
        started.browser_nonce,
        max_age=int(SIGNIN_TTL.total_seconds()),
        path=_COOKIE_PATH,
        httponly=True,
        # Lax is sent on GitHub's top-level redirect back, and on nothing cross-site besides.
        samesite="lax",
        secure=services.secure_cookies,
    )
    return response


@router.get(CALLBACK_PATH)
async def finish_sign_in(
    services: ServicesDep,
    browser_nonce: Annotated[str | None, Cookie(alias=_NONCE_COOKIE)] = None,
    state: str = "",
    code: str | None = None,
    error: str | None = None,
) -> Response:
    try:
        target = await services.signin.finish(
            state=state, browser_nonce=browser_nonce, code=code, error=error
        )
    except SignInError as exc:
        return _browser_error(exc)

    response = RedirectResponse(target, status_code=302)
    response.delete_cookie(_NONCE_COOKIE, path=_COOKIE_PATH)
    return response


@router.post("/auth/token")
async def issue_tokens(
    body: TokenRequest, services: ServicesDep, response: Response
) -> TokenResponse:
    # Tokens must never sit in a cache between the API and the app.
    response.headers["Cache-Control"] = "no-store"
    try:
        if body.grant_type == "authorization_code":
            if not body.code or not body.code_verifier:
                raise SignInError("invalid_request", "code and code_verifier are required.")
            pair = await services.signin.redeem(code=body.code, code_verifier=body.code_verifier)
        else:
            if not body.refresh_token:
                raise SignInError("invalid_request", "refresh_token is required.")
            pair = await services.signin.refresh(body.refresh_token)
    except SignInError as exc:
        raise HTTPException(
            exc.status,
            detail={"code": exc.code, "message": exc.message},
            headers={"Cache-Control": "no-store"},
        ) from None

    return TokenResponse(
        access_token=pair.access_token,
        expires_in=pair.expires_in,
        refresh_token=pair.refresh_token,
    )


@router.post("/auth/logout", status_code=204)
async def sign_out(body: SignOutRequest, services: ServicesDep) -> Response:
    await services.signin.sign_out(body.refresh_token)
    return Response(status_code=204)


@router.get("/me")
async def me(user: CurrentUser, services: ServicesDep) -> MeResponse:
    record = await services.accounts.get_user(user.user_id)
    if record is None:
        raise unauthorized("not_signed_in", "This account no longer exists.")
    return MeResponse(id=record.id, login=record.login, avatar_url=record.avatar_url)


def _browser_error(error: SignInError) -> HTMLResponse:
    page = (
        "<!doctype html><meta charset=utf-8><title>InterroAI sign-in</title>"
        f"<h1>Sign-in failed</h1><p>{html.escape(error.message)}</p>"
    )
    return HTMLResponse(page, status_code=error.status)
