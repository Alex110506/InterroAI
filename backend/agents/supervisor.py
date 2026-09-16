"""
Central Hierarchical Orchestrator — Supervisor / Manager.

Receives the routed prompt from the chat session and drives the
Coding Agent's Plan → Code → Verify cycle.

`stream()` is the transport-agnostic form and is what the chat session uses.
`run()` wraps it for a WebSocket, and stops pushing as soon as the socket
refuses an event — the client has gone, and the remaining events would each
raise for no benefit.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.gateway import ModelGateway

logger = logging.getLogger(__name__)


async def stream(
    prompt: str,
    project_path: str,
    model: str,
    intent: str = "implement",
    history: list[dict] | None = None,
    gateway: ModelGateway | None = None,
    effort: str | None = None,
) -> AsyncIterator[dict]:
    """Run the Coding Agent, yielding each of its events in order."""
    from agents.coder import CoderAgent

    logger.info(
        "Task starting | model=%r | effort=%r | project=%r",
        model,
        effort,
        Path(project_path).name,
    )

    agent = CoderAgent(
        project_path=project_path,
        model=model,
        intent=intent,
        history=history,
        gateway=gateway,
        effort=effort,
    )
    async for event in agent.execute(prompt):
        logger.info("coder event: %s", event.get("type"))
        yield event

    logger.info("Task complete | model=%r", model)


async def run(
    prompt: str,
    project_path: str,
    model: str,
    websocket=None,
    intent: str = "implement",
    history: list[dict] | None = None,
    gateway: ModelGateway | None = None,
    effort: str | None = None,
) -> None:
    """Drive `stream()` and relay every event over *websocket*, if given."""
    async for event in stream(
        prompt,
        project_path,
        model,
        intent=intent,
        history=history,
        gateway=gateway,
        effort=effort,
    ):
        if websocket is not None:
            try:
                await websocket.send_json(event)
            except Exception:
                break
