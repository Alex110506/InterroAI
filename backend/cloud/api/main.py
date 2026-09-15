"""
The Web API's application factory.

Run it from `backend/` with:

    uvicorn cloud.api.main:create_app --factory --port 8080

A factory rather than a module-level `app`, so nothing — settings, the database
engine, Azure clients — is constructed at import time. Tests build an app with
their own settings and fakes, and importing this module never needs a `.env`.
"""
from __future__ import annotations

from fastapi import FastAPI

VERSION = "0.1.0"


def create_app() -> FastAPI:
    app = FastAPI(title="InterroAI Cloud API", version=VERSION)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": VERSION}

    return app
