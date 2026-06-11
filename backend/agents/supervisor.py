"""
Central Hierarchical Orchestrator — Supervisor / Manager.

Receives an enriched, fully-specified prompt from the Grill agent and
coordinates the execution pipeline:
  1. Delegates to the Router to select a model tier.
  2. Passes the task to the Coder agent (Plan → Code → Verify).
"""

from __future__ import annotations


async def run(prompt: str, project_path: str, model: str) -> None:
    """Orchestrate the full task pipeline for a given prompt and workspace.
    Currently a stub that just represents passing the task to the Coder Agent.
    """
    pass
