"""
The Web API's application factory.

Run it from `backend/` with:

    uvicorn cloud.api.main:create_app --factory --port 8080

A factory rather than a module-level `app`, so nothing — settings, the database
engine, Azure clients — is constructed at import time, and importing this module
never needs a `.env`. Tests hand in their own `Services`. Without them the app
builds its own from `ApiSettings` as it starts, so a missing secret stops the
process at startup rather than failing its first request.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cloud.api import auth, indexing, llm_gateway, projects
from cloud.api.middleware import RequestContextMiddleware
from cloud.api.services import Services, build_services
from cloud.observability import configure_logging
from cloud.settings import ApiSettings

VERSION = "0.1.0"


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            yield
            return
        settings = ApiSettings()
        configure_logging(log_format=settings.log_format, level=settings.log_level)
        built = build_services(settings)
        app.state.services = built
        try:
            yield
        finally:
            await built.close()

    app = FastAPI(title="InterroAI Cloud API", version=VERSION, lifespan=lifespan)
    if services is not None:
        app.state.services = services
    app.add_middleware(RequestContextMiddleware)

    app.include_router(auth.router)
    app.include_router(projects.router)
    app.include_router(indexing.router)
    app.include_router(llm_gateway.router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": VERSION}

    return app
