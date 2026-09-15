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

from cloud.api import auth, projects
from cloud.api.services import Services, build_services
from cloud.settings import ApiSettings

VERSION = "0.1.0"


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            yield
            return
        built = build_services(ApiSettings())
        app.state.services = built
        try:
            yield
        finally:
            await built.close()

    app = FastAPI(title="InterroAI Cloud API", version=VERSION, lifespan=lifespan)
    if services is not None:
        app.state.services = services

    app.include_router(auth.router)
    app.include_router(projects.router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": VERSION}

    return app
