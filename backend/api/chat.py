"""
Chat WebSocket endpoint.

Every message goes through a pipeline:

  Step 1 — Model Selection:
    If 'auto' is selected, call the router to determine the best model.
    Otherwise, use the user's selected model.

  Step 2 — Intent classification:
    gpt-4o-mini evaluates if the user's prompt is ambiguous and needs requirement
    clarification before implementation.

  Step 3 — Route:
    • Interrogation Needed (Route A): 
        Call GrillAgent to refine the prompt in one shot. Send refined prompt
        to the Coding Agent.
    • Interrogation Not Needed (Route B):
        Skip GrillAgent. Send raw prompt directly to the Coding Agent.
"""
from __future__ import annotations

import json
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from agents.grill import GrillAgent, _fmt_git, _fmt_tree
from agents import router as model_router
from agents import supervisor
from core.security import retrieve_openai_key
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/chat", tags=["chat"])

_CLASSIFIER_SYSTEM = """\
You are a senior software engineering manager. A user has submitted a request to build or modify a codebase.
Evaluate if the request is ambiguous and requires technical clarification before a developer can implement it.
If the request is clear enough, or if it is just a general question/explanation, no interrogation is needed.

Respond strictly with JSON:
{"needs_interrogation": true} or {"needs_interrogation": false}
"""


async def _needs_interrogation(user_message: str, project_index: dict) -> bool:
    """
    Classify whether the user's prompt needs refinement.
    """
    key = retrieve_openai_key()
    if not key:
        return False

    tree_str = _fmt_tree(project_index.get("file_tree") or {})
    git_str = _fmt_git(project_index.get("git_context") or {})
    context = f"PROJECT STRUCTURE:\n{tree_str or '(empty)'}\n\nGIT CONTEXT:\n{git_str}"

    client = AsyncOpenAI(api_key=key)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{_CLASSIFIER_SYSTEM}\n\n{context}"},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return data.get("needs_interrogation", False)
    except Exception:
        return False


# ── WebSocket handler ─────────────────────────────────────────────────────────

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    agent: GrillAgent | None = None
    turn = 0
    active_project_path = ""
    active_selected_model = "auto"

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "start":
                project_path = payload.get("project_path", "")
                active_project_path = project_path
                project_index = payload.get("project_index") or {}
                user_message = payload.get("message", "")
                selected_model = payload.get("model", "auto")

                # Step 1: Model Selection
                if selected_model == "auto":
                    logger.info("Model selection set to 'auto'. Evaluating prompt complexity...")
                    selected_model = await model_router.classify(user_message)
                    logger.info(f"Automated router selected model: {selected_model}")
                else:
                    logger.info(f"User manually selected model: {selected_model}")
                
                active_selected_model = selected_model

                # Step 2: Intent classification
                needs_interrogation = await _needs_interrogation(user_message, project_index)

                # Step 3: Route
                if needs_interrogation:
                    agent = GrillAgent(project_path=project_path, project_index=project_index)
                    result = await agent.start(user_message)

                    if result.get("is_prompt_ready"):
                        final_prompt = result.get("refined_prompt", "")
                        await websocket.send_json({
                            "type": "ready",
                            "refined_prompt": final_prompt,
                            "did_interrogate": turn > 0,
                        })
                        await supervisor.run(final_prompt, active_project_path, active_selected_model)
                        break
                    
                    turn += 1
                    await websocket.send_json({
                        "type": "question",
                        "question": result.get("question", ""),
                        "turn": turn,
                    })
                else:
                    final_prompt = user_message
                    await websocket.send_json({
                        "type": "ready",
                        "refined_prompt": final_prompt,
                        "did_interrogate": False,
                    })
                    await supervisor.run(final_prompt, active_project_path, active_selected_model)
                    break

            # ── User answered a clarifying question ───────────────────────────
            elif msg_type == "answer":
                if agent is None:
                    await websocket.send_json({"type": "error", "message": "No active session."})
                    continue
                result = await agent.answer(payload.get("message", ""))

                if result.get("is_prompt_ready"):
                    final_prompt = result.get("refined_prompt", "")
                    await websocket.send_json({
                        "type": "ready",
                        "refined_prompt": final_prompt,
                        "did_interrogate": turn > 0,
                    })
                    await supervisor.run(final_prompt, active_project_path, active_selected_model)
                    break
                
                turn += 1
                await websocket.send_json({
                    "type": "question",
                    "question": result.get("question", ""),
                    "turn": turn,
                })

            # ── Cancel button → force-generate refined prompt ─────────────────
            elif msg_type == "force_ready":
                if agent is None:
                    await websocket.send_json({"type": "error", "message": "No active session."})
                    continue
                result = await agent._force_ready()
                final_prompt = result.get("refined_prompt", "")
                await websocket.send_json({
                    "type": "ready",
                    "refined_prompt": final_prompt,
                    "did_interrogate": turn > 0,
                })
                await supervisor.run(final_prompt, active_project_path, active_selected_model)
                break

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

