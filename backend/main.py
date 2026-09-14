"""
InterroAI — the local runtime's HTTP surface.

The Electron app (`frontend/`) is the frontend. It drives the pipeline through
this process over HTTP + WebSocket on 127.0.0.1:8000; `npm run dev` in
`frontend/` starts both.

Run it on its own with:
    interroai-backend
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.projects import router as projects_router
from api.settings import router as settings_router

app = FastAPI(
    title="InterroAI",
    description="AI Multi-Agent Orchestration Platform",
    version="0.1.0",
)

# Permissive by design: the server binds to 127.0.0.1 and holds one local
# user's workspace, so the origin of a request carries no authority worth
# checking. Anything that can reach the port is already on the machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(settings_router)
app.include_router(projects_router)
app.include_router(chat_router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "version": app.version}


def run() -> None:
    """Console-script entry point (`interroai-backend`) declared in pyproject.toml."""
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
