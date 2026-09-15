"""
FastAPI dependencies and helpers shared by the routers.

Errors meant for the app carry a stable `code` in `detail`, so the runtime can
act on them without parsing the message, which is for people: refresh on
`invalid_token`, show the sign-in screen on `not_signed_in`, tell the user to
wait on `quota_exceeded`.
"""
from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
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


def api_error(status_code: int, code: str, message: str, **extra: object) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message, **extra})


def unauthorized(code: str, message: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail={"code": code, "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def charge_request(services: Services, user: AccessClaims) -> None:
    """Count one request that spends the platform key, or refuse it with a 429."""
    now = datetime.now(UTC)
    if await services.usage.admit(user.user_id, now.date(), services.limits.quota):
        return
    midnight = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=UTC)
    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "code": "quota_exceeded",
            "message": "Today's allowance is used up. It resets at midnight UTC.",
        },
        headers={"Retry-After": str(int((midnight - now).total_seconds()) + 1)},
    )


ServicesDep = Annotated[Services, Depends(get_services)]
CurrentUser = Annotated[AccessClaims, Depends(current_user)]
