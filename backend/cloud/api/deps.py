"""
FastAPI dependencies and helpers shared by the routers.

Errors meant for the app carry a stable `code` in `detail`, so the runtime can
act on them without parsing the message, which is for people: refresh on
`invalid_token`, show the sign-in screen on `not_signed_in` or `access_denied`,
tell the user to wait on `quota_exceeded` or `rate_limited`.
"""
from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator
from datetime import UTC, datetime, time, timedelta
from typing import Annotated

import openai
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cloud import observability
from cloud.api.services import Services
from cloud.auth.auth0 import Auth0Unavailable, InvalidTokenError
from cloud.auth.identity import AccessClaims, AccessDeniedError
from core.errors import MissingAPIKeyError

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


async def get_services(request: Request) -> Services:
    return request.app.state.services


async def bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None:
        raise unauthorized("not_signed_in", "Sign in to use InterroAI.")
    return credentials.credentials


async def current_user(
    services: Annotated[Services, Depends(get_services)],
    token: Annotated[str, Depends(bearer_token)],
) -> AccessClaims:
    # Async on purpose: FastAPI runs plain functions in a worker thread, and the
    # user id set below would not reach this request's access log line.
    try:
        claims = await services.identity.resolve(token)
    except InvalidTokenError:
        raise unauthorized("invalid_token", "The access token is invalid or has expired.") from None
    except AccessDeniedError as exc:
        raise api_error(403, "access_denied", str(exc)) from None
    except Auth0Unavailable:
        logger.warning("Auth0 was unreachable while checking a request", exc_info=True)
        raise api_error(
            503, "auth_unavailable", "Sign-in is unavailable right now. Try again shortly."
        ) from None
    observability.user_id.set(claims.user_id)
    return claims


def api_error(status_code: int, code: str, message: str, **extra: object) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message, **extra})


@contextlib.contextmanager
def upstream_errors() -> Iterator[None]:
    """OpenAI's failures, as answers the runtime can act on."""
    try:
        yield
    except MissingAPIKeyError:
        logger.error("The platform OpenAI key is not configured")
        raise api_error(503, "llm_unavailable", "The model service is not configured.") from None
    except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
        raise api_error(400, "invalid_request", exc.message) from None
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        logger.error("OpenAI refused the platform key")
        raise api_error(503, "llm_unavailable", "The model service is unavailable.") from None
    except openai.APITimeoutError:
        # Deliberately not retried (cloud.llm.openai_client), and Container Apps
        # cancels the request at 240 seconds anyway. Saying so plainly matters:
        # the runtime would otherwise report the whole cloud as unreachable.
        raise api_error(
            504,
            "model_timeout",
            "The model took too long to answer. Try a smaller task, or a faster model.",
        ) from None
    except (openai.RateLimitError, openai.APIConnectionError):
        # Already retried with backoff in cloud.llm.openai_client.
        raise api_error(
            503, "llm_unavailable", "The model service is busy. Try again shortly."
        ) from None
    except openai.OpenAIError:
        logger.warning("OpenAI request failed", exc_info=True)
        raise api_error(502, "upstream_error", "The model service failed.") from None


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
BearerToken = Annotated[str, Depends(bearer_token)]
