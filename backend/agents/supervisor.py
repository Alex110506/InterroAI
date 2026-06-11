"""
Central Hierarchical Orchestrator — Supervisor / Manager.

Receives the final refined prompt from the Grill agent and drives the
Coding Agent's Plan → Code → Verify cycle, streaming every event back
through the WebSocket.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run(prompt: str, project_path: str, model: str, websocket=None) -> None:
    from agents.coder import CoderAgent
    from pathlib import Path

    logger.info("Task starting | model=%r | project=%r", model, Path(project_path).name)

    agent = CoderAgent(project_path=project_path, model=model)
    async for event in agent.execute(prompt):
        logger.info("coder event: %s", event.get("type"))
        if websocket is not None:
            try:
                await websocket.send_json(event)
            except Exception:
                break

    logger.info("Task complete | model=%r", model)
