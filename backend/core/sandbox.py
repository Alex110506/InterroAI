"""
Sandbox: linting and test execution for the Coding Agent's validation phase.
Tools that are not installed are silently skipped (no crash, just pass-through).
"""
from __future__ import annotations

import asyncio
from pathlib import Path


async def run_linter(file_paths: list[str], project_path: str) -> tuple[bool, str]:
    """Run ruff over modified Python files. Silent pass if ruff is not installed."""
    py_files = [f for f in file_paths if f.endswith(".py") and Path(f).exists()]
    if not py_files:
        return True, ""
    return await _exec(["ruff", "check", "--output-format=text", *py_files], project_path)


async def run_tests(project_path: str) -> tuple[bool, str]:
    """Run pytest. Skips gracefully if pytest is absent or no test files exist."""
    root = Path(project_path)
    has_tests = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py"))
    if not has_tests:
        return True, "No test files found — skipped."
    return await _exec(["pytest", "--tb=short", "-q"], project_path)


async def _exec(cmd: list[str], cwd: str) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = (stdout + stderr).decode(errors="replace").strip()
        return proc.returncode == 0, output
    except FileNotFoundError:
        return True, ""          # tool not installed — silent pass
    except asyncio.TimeoutError:
        return False, "Timeout after 60 s"
