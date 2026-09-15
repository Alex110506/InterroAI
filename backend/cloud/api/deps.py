"""
FastAPI dependencies and helpers shared by the routers.

Errors meant for the app carry a stable `code` in `detail`, so the runtime can
act on them without parsing the message, which is for people: refresh on
`invalid_token`, show the sign-in screen on `not_signed_in`, tell the user to
wait on `quota_exceeded` or `rate_limited`.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, time, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cloud import observability
from cloud.api.services import Services
from cloud.api.tokens import AccessClaims, InvalidTokenError

_bearer = HTTPBearer(auto_error=False)


async def get_services(request: Request) -> Services:
    return request.app.state.services


async def current_user(
    services: Annotated[Services, Depends(get_services)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AccessClaims:
    # Async on purpose: FastAPI runs plain functions in a worker thread, and the
    # user id set below would not reach this request's access log line.
    if credentials is None:
        raise unauthorized("not_signed_in", "Sign in to use InterroAI.")
    try:
        claims = services.signer.read_access_token(credentials.credentials)
    except InvalidTokenError:
        raise unauthorized("invalid_token", "The access token is invalid or has expired.") from None
    observability.user_id.set(claims.user_id)
    return claims


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


def rate_limit(limit: str):
    """A dependency that authenticates the caller and counts the request against *limit*."""

    async def check(
        user: Annotated[AccessClaims, Depends(current_user)],
        services: Annotated[Services, Depends(get_services)],
    ) -> AccessClaims:
        _spend(services, limit, f"user:{user.user_id}")
        return user

    return check


def rate_limit_by_address(limit: str):
    """A dependency that counts the request against *limit* per client address, for sign-in."""

    async def check(request: Request, services: Annotated[Services, Depends(get_services)]) -> None:
        address = request.client.host if request.client else "unknown"
        _spend(services, limit, f"address:{address}")

    return check


def _spend(services: Services, limit: str, caller: str) -> None:
    wait = services.limiter.hit(limit, caller, getattr(services.limits.rates, limit))
    if wait:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limited",
                "message": "Too many requests at once. Try again in a moment.",
            },
            headers={"Retry-After": str(math.ceil(wait))},
        )


ServicesDep = Annotated[Services, Depends(get_services)]
CurrentUser = Annotated[AccessClaims, Depends(current_user)]
