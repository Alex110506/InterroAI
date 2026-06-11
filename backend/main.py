"""
InterroAI — FastAPI backend.
Async server and WebSocket connection management (Section 6 of architecture spec).

Run in development:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.settings import router as settings_router
from api.projects import router as projects_router
from api.chat import router as chat_router

app = FastAPI(
    title="InterroAI",
    description="AI Multi-Agent Orchestration Platform",
    version="0.1.0",
)

# Allow requests from the Electron renderer in both dev and production modes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "app://.",                # Electron production (custom protocol)
        "file://",                # Electron production fallback
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(settings_router)
app.include_router(projects_router)
app.include_router(chat_router)

# Future routers go here:
# app.include_router(agent_router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "version": app.version}
