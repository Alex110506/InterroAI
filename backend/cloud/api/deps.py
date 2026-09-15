"""
FastAPI dependencies shared by the routers: the process's services, and the caller.

Errors meant for the app carry a stable `code` in `detail`, so the runtime can
act on them (refresh on `invalid_token`, show a sign-in screen on
`not_signed_in`) without parsing the message, which is for people.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cloud.api.services import Services
from cloud.api.tokens import AccessClaims, InvalidTokenError

_bearer = HTTPBearer(auto_error=False)


def get_services(request: Request) -> Services:
    return request.app.state.services


def current_user(
    services: Annotated[Services, Depends(get_services)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AccessClaims:
    if credentials is None:
        raise unauthorized("not_signed_in", "Sign in to use InterroAI.")
    try:
        return services.signer.read_access_token(credentials.credentials)
    except InvalidTokenError:
        raise unauthorized("invalid_token", "The access token is invalid or has expired.") from None


def unauthorized(code: str, message: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail={"code": code, "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


ServicesDep = Annotated[Services, Depends(get_services)]
CurrentUser = Annotated[AccessClaims, Depends(current_user)]
