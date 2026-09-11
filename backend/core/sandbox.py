"""
Sandbox: linting and test execution for the Coding Agent's validation phase.

A check has three outcomes, not two. The previous version collapsed "ruff isn't
installed" into the same `(True, "")` that meant "ruff ran and found nothing
wrong", so a project with no linter available reported a clean bill of health it
had never actually earned. `CheckStatus.SKIPPED` keeps that distinction visible
all the way to the UI.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"   # the check could not run; it did NOT succeed


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    output: str

    @property
    def ok(self) -> bool:
        """
        True when the check does not block the agent.

        A skipped check is non-blocking — we cannot ask the model to fix errors
        from a tool that never ran — but callers that want to report honestly
        should branch on `status`, not on this.
        """
        return self.status is not CheckStatus.FAILED


async def run_linter(file_paths: list[str], project_path: str) -> CheckResult:
    """Run ruff over modified Python files."""
    py_files = [f for f in file_paths if f.endswith(".py") and Path(f).exists()]
    if not py_files:
        return CheckResult(CheckStatus.SKIPPED, "No Python files were modified.")
    return await _exec(
        ["ruff", "check", "--output-format=text", *py_files],
        project_path,
        tool="ruff",
    )


async def run_tests(project_path: str) -> CheckResult:
    """Run pytest, if the project has any test files."""
    root = Path(project_path)
    has_tests = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py"))
    if not has_tests:
        return CheckResult(CheckStatus.SKIPPED, "No test files found.")
    return await _exec(["pytest", "--tb=short", "-q"], project_path, tool="pytest")


async def _exec(cmd: list[str], cwd: str, *, tool: str) -> CheckResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT_SECONDS
        )
    except FileNotFoundError:
        # Expected: the user's project environment simply has no such tool.
        logger.info("%s is not installed — skipping this check.", tool)
        return CheckResult(
            CheckStatus.SKIPPED, f"{tool} is not installed in this environment."
        )
    except TimeoutError:
        logger.warning("%s exceeded %ss and was killed.", tool, _TIMEOUT_SECONDS)
        proc.kill()
        return CheckResult(
            CheckStatus.FAILED, f"{tool} timed out after {_TIMEOUT_SECONDS}s."
        )
    except OSError:
        # Unexpected: the tool exists but could not be spawned. Keep the stack
        # trace rather than reporting a false pass.
        logger.exception("Failed to spawn %s.", tool)
        return CheckResult(CheckStatus.FAILED, f"Could not run {tool} — see backend logs.")

    output = (stdout + stderr).decode(errors="replace").strip()
    status = CheckStatus.PASSED if proc.returncode == 0 else CheckStatus.FAILED
    return CheckResult(status, output)
