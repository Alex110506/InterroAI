"""
Source Code Writer and Modifier (Section 5 of architecture spec).

Enforces a strict Plan → Code → Verify cycle:

  Phase A — Planning:
    Generate a human-readable attack plan before touching any file.

  Phase B — Modification via Unified Diff Blocks:
    Return SEARCH / REPLACE blocks only — no full-file rewrites.
    The backend applies the diffs programmatically.

  Phase C — Verification Loop (up to 3 auto-correction attempts):
    1. Run linter (static analysis) on modified files.
    2. Execute the project's test suite (pytest / equivalent).
    3. On failure, feed stderr back to the model for self-correction.
"""

from __future__ import annotations


async def execute(enriched_prompt: str, project_path: str) -> None:
    """Run the Plan → Code → Verify pipeline for the given task."""
    raise NotImplementedError
