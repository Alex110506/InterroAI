"""
InterroAI — the local runtime's HTTP surface.

The Electron app (`frontend/`) is the frontend. It starts this process itself,
on a free port on 127.0.0.1 with a random launch token (`api/guard.py`), and
drives the pipeline over HTTP + WebSocket. `npm run dev` in `frontend/` starts
the app, and the app starts this.

Run it on its own, for development, with:
    interroai-backend
    uvicorn main:app --reload --port 8000
Started that way it has no launch token, and answers anything that can reach
the port.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.guard import TOKEN_HEADER, LaunchTokenMiddleware
from api.projects import router as projects_router
from api.session import router as session_router
from api.settings import router as settings_router
from core.settings import RuntimeSettings, get_runtime_settings


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    settings = settings or get_runtime_settings()
    app = FastAPI(
        title="InterroAI",
        description="AI Multi-Agent Orchestration Platform",
        version="0.1.0",
    )

    if settings.launch_token is not None:
        app.add_middleware(
            LaunchTokenMiddleware, token=settings.launch_token.get_secret_value()
        )
    # Added last, so it runs first: it answers CORS preflights, which carry no
    # token, and it puts CORS headers on the token middleware's refusals too, so
    # the app can read why it was turned away.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list,
        allow_methods=["*"],
        allow_headers=["content-type", TOKEN_HEADER],
    )

    app.include_router(settings_router)
    app.include_router(session_router)
    app.include_router(projects_router)
    app.include_router(chat_router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": app.version}

    return app


app = create_app()


def run() -> None:
    """Console-script entry point (`interroai-backend`) declared in pyproject.toml."""
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
