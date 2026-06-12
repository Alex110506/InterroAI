"""
Chat WebSocket endpoint.

Every message goes through a pipeline:

  Step 1 — Intent classification (3-way):
    gpt-5.4-mini classifies the user's request as one of:
      • "answer"      — general question; reply directly with Markdown
      • "interrogate" — ambiguous implementation request; use GrillAgent
      • "implement"   — clear implementation task; go straight to coder

  Step 2 — Model Selection (for implement/interrogate paths):
    If 'auto' is selected, call the router to determine the best model.

  Step 3 — Route:
    • answer:      send message event, close
    • interrogate: GrillAgent → refine → coder
    • implement:   coder directly
"""
from __future__ import annotations

import json
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from agents.grill import GrillAgent, _fmt_git, _fmt_tree
from agents import router as model_router
from agents import supervisor
from agents.coder import _MODEL_MAP as _CODER_MODEL_MAP, _REASONING_MODELS
from core.security import retrieve_openai_key
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/chat", tags=["chat"])

_INTENT_SYSTEM = """\
You are a routing agent for an AI coding assistant. Classify the user's request into exactly one of three actions:

1. "answer" — The user is asking a general question, requesting an explanation, or having a conversation.
   They are NOT asking to build, change, fix, or implement anything in the codebase.

2. "interrogate" — The user wants to implement/build/fix/modify something in the codebase,
   BUT the request is vague or ambiguous and needs clarification before a developer can act on it.

3. "implement" — The user wants to implement/build/fix/modify something in the codebase,
   AND the request is already specific and clear enough for a developer to act on without further questions.

Respond strictly with JSON: {"action": "answer"} or {"action": "interrogate"} or {"action": "implement"}
"""

_QA_SYSTEM = """\
You are a helpful AI assistant for a software project. Answer the user's question concisely and clearly in Markdown.
Use the project context below to inform your answer.
"""

# Q&A answers now use the main _CODER_MODEL_MAP so the user's selected large model
# is actually used, handling reasoning model constraints (no temperature).


async def _classify_intent(user_message: str, project_index: dict) -> str:
    """3-way intent classifier. Returns: 'answer' | 'interrogate' | 'implement'."""
    key = retrieve_openai_key()
    if not key:
        return "implement"

    tree_str = _fmt_tree(project_index.get("file_tree") or {})
    git_str = _fmt_git(project_index.get("git_context") or {})
    context = f"PROJECT STRUCTURE:\n{tree_str or '(empty)'}\n\nGIT CONTEXT:\n{git_str}"

    client = AsyncOpenAI(api_key=key)
    try:
        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": f"{_INTENT_SYSTEM}\n\n{context}"},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        action = data.get("action", "implement")
        if action not in ("answer", "interrogate", "implement"):
            action = "implement"
        return action
    except Exception:
        return "implement"


# _answer_question has been removed. We now route Q&A directly to the CoderAgent
# so it has access to file reading and search tools.


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
                active_selected_model = selected_model  # may still be "auto" until final prompt
                if selected_model != "auto":
                    logger.info(f"User manually selected model: {selected_model}")

                # Step 1: Intent classification (3-way)
                action = await _classify_intent(user_message, project_index)
                logger.info(f"Intent classified: {action!r}")

                # Step 2+3: Route
                if action == "answer":
                    if active_selected_model == "auto":
                        active_selected_model = await model_router.classify(user_message)
                        logger.info(f"Auto-router selected (Q&A): {active_selected_model}")
                    
                    await websocket.send_json({
                        "type": "ready",
                        "refined_prompt": user_message,
                        "did_interrogate": False,
                    })
                    logger.info(f"Dispatching to Q&A agent | model={active_selected_model!r}")
                    await supervisor.run(user_message, active_project_path, active_selected_model, websocket, intent="answer")
                    break

                elif action == "interrogate":
                    agent = GrillAgent(project_path=project_path, project_index=project_index)
                    result = await agent.start(user_message)

                    if result.get("is_prompt_ready"):
                        final_prompt = result.get("refined_prompt", "")
                        if active_selected_model == "auto":
                            active_selected_model = await model_router.classify(final_prompt)
                            logger.info(f"Auto-router selected: {active_selected_model}")
                        await websocket.send_json({
                            "type": "ready",
                            "refined_prompt": final_prompt,
                            "did_interrogate": turn > 0,
                        })
                        logger.info(f"Dispatching to coder | project={active_project_path!r} | model={active_selected_model!r}")
                        await supervisor.run(final_prompt, active_project_path, active_selected_model, websocket)
                        break

                    turn += 1
                    await websocket.send_json({
                        "type": "question",
                        "question": result.get("question", ""),
                        "turn": turn,
                    })

                else:  # "implement"
                    final_prompt = user_message
                    if active_selected_model == "auto":
                        active_selected_model = await model_router.classify(final_prompt)
                        logger.info(f"Auto-router selected: {active_selected_model}")
                    await websocket.send_json({
                        "type": "ready",
                        "refined_prompt": final_prompt,
                        "did_interrogate": False,
                    })
                    logger.info(f"Dispatching to coder | model={active_selected_model!r}")
                    await supervisor.run(final_prompt, active_project_path, active_selected_model, websocket)
                    break

            # ── User answered a clarifying question ───────────────────────────
            elif msg_type == "answer":
                if agent is None:
                    await websocket.send_json({"type": "error", "message": "No active session."})
                    continue
                result = await agent.answer(payload.get("message", ""))

                if result.get("is_prompt_ready"):
                    final_prompt = result.get("refined_prompt", "")
                    if active_selected_model == "auto":
                        active_selected_model = await model_router.classify(final_prompt)
                        logger.info(f"Auto-router selected: {active_selected_model}")
                    await websocket.send_json({
                        "type": "ready",
                        "refined_prompt": final_prompt,
                        "did_interrogate": turn > 0,
                    })
                    await supervisor.run(final_prompt, active_project_path, active_selected_model, websocket)
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
                if active_selected_model == "auto":
                    active_selected_model = await model_router.classify(final_prompt)
                    logger.info(f"Auto-router selected: {active_selected_model}")
                await websocket.send_json({
                    "type": "ready",
                    "refined_prompt": final_prompt,
                    "did_interrogate": turn > 0,
                })
                await supervisor.run(final_prompt, active_project_path, active_selected_model, websocket)
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

