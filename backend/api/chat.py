"""
Chat WebSocket endpoint.

A transport adapter, nothing more: every frame is turned into a call on a
`ChatSession` (`agents/session.py`) and every event the session emits is
relayed back as JSON. The pipeline itself — intent classification, dispatch
to the coder — lives in the session, so it can be driven and tested without a
socket.

Client frames:
  {"type": "start",       "project_path", "project_index", "message", "model",
                          "history"}       — prior turns, oldest first, in
                                             OpenAI message shape. Optional;
                                             the server stores no transcript,
                                             so a client that wants follow-ups
                                             to resolve must send its own.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agents.session import _DEFAULT_MODEL, ChatSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

__all__ = ["router", "_DEFAULT_MODEL"]


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type != "start":
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )
                continue

            session = ChatSession(
                project_path=payload.get("project_path", ""),
                project_index=payload.get("project_index") or {},
                model=payload.get("model"),
                history=payload.get("history"),
            )

            async for event in session.start(payload.get("message", "")):
                await websocket.send_json(event)

            if session.finished:
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        # The session converts its own failures into `error` events, so
        # reaching here means the socket itself misbehaved.
        logger.exception("Unhandled error in chat WebSocket")
        try:
            await websocket.send_json({"type": "error", "message": f"Internal error: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
