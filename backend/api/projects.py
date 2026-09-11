"""
Project indexing HTTP/WebSocket surface.

A transport adapter over `core/project_index.py`, which owns the actual
scanning, chunking, embedding and storage:

  Phase 1 (POST /index):          file tree + git context → ProjectIndex JSON.
  Phase 2 (WebSocket /ws/embed):  Steps A-D streamed as live progress events.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.project_index import ProjectIndex, ProjectPathError, build_index, embed_project

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


class IndexRequest(BaseModel):
    path: str


@router.post("/index", response_model=ProjectIndex)
async def index_project(body: IndexRequest) -> ProjectIndex:
    """Index a project folder (Phase 1)."""
    try:
        return build_index(body.path)
    except ProjectPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/ws/embed")
async def embed_project_ws(websocket: WebSocket) -> None:
    """
    Phase 2 RAG indexing pipeline (Sections 3A-D).

    Client sends:  { "path": "/abs/path/to/project" }
    Server emits a stream of JSON progress events:
      { "step": "A"|"B"|"C"|"D", "status": "start"|"done"|"progress", ...fields }
      { "step": "done" }
      { "step": "error", "message": "..." }
    """
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        async for event in embed_project(payload["path"]):
            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        # `embed_project` reports its own failures as error events, so reaching
        # here means the socket or the opening frame was at fault.
        logger.exception("Unhandled error on the indexing socket")
        try:
            await websocket.send_json({"step": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
