"""
The sandbox's job is to report honestly. The bug worth guarding against is a
check that never ran being reported as a check that passed.
"""
from __future__ import annotations

import asyncio

import pytest

import core.workspace.sandbox as sandbox
from core.workspace.sandbox import CheckResult, CheckStatus, run_linter, run_tests

# ── CheckResult semantics ────────────────────────────────────────────────────


def test_passed_is_ok():
    assert CheckResult(CheckStatus.PASSED, "").ok is True


def test_failed_is_not_ok():
    assert CheckResult(CheckStatus.FAILED, "err").ok is False


def test_skipped_is_non_blocking_but_distinguishable():
    """
    `ok` is True because the agent cannot fix errors from a tool that never
    ran — but `status` still says SKIPPED, so the UI never prints "passed".
    """
    result = CheckResult(CheckStatus.SKIPPED, "ruff is not installed")
    assert result.ok is True
    assert result.status is CheckStatus.SKIPPED
    assert result.status.value == "skipped"


def test_status_serialises_as_a_plain_string():
    """The value is sent over the WebSocket as JSON, so it must be a str enum."""
    assert isinstance(CheckStatus.SKIPPED.value, str)
    assert CheckStatus.PASSED.value == "passed"
    assert CheckStatus.FAILED.value == "failed"


def test_result_is_immutable():
    result = CheckResult(CheckStatus.PASSED, "")
    with pytest.raises(Exception):  # noqa: B017
        result.status = CheckStatus.FAILED


# ── _exec: the subprocess boundary ───────────────────────────────────────────


async def test_exec_reports_success(tmp_path):
    result = await sandbox._exec(["sh", "-c", "exit 0"], str(tmp_path), tool="sh")
    assert result.status is CheckStatus.PASSED


async def test_exec_reports_failure_with_output(tmp_path):
    result = await sandbox._exec(
        ["sh", "-c", "echo boom >&2; exit 1"], str(tmp_path), tool="sh"
    )
    assert result.status is CheckStatus.FAILED
    assert "boom" in result.output


async def test_exec_merges_stdout_and_stderr(tmp_path):
    result = await sandbox._exec(
        ["sh", "-c", "echo out; echo err >&2; exit 1"], str(tmp_path), tool="sh"
    )
    assert "out" in result.output and "err" in result.output


async def test_missing_tool_is_skipped_not_passed(tmp_path):
    """
    The core regression: a tool that isn't installed used to return
    `(True, "")`, indistinguishable from a clean run.
    """
    result = await sandbox._exec(
        ["interroai-not-a-real-binary"], str(tmp_path), tool="ghost"
    )
    assert result.status is CheckStatus.SKIPPED
    assert "not installed" in result.output


async def test_timeout_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "_TIMEOUT_SECONDS", 0.2)
    result = await sandbox._exec(["sh", "-c", "sleep 5"], str(tmp_path), tool="sh")
    assert result.status is CheckStatus.FAILED
    assert "timed out" in result.output


async def test_timeout_kills_the_child_process(tmp_path, monkeypatch):
    """A timed-out check must not leave the subprocess running."""
    monkeypatch.setattr(sandbox, "_TIMEOUT_SECONDS", 0.2)
    killed = {}
    real_exec = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        proc = await real_exec(*args, **kwargs)
        killed["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    await sandbox._exec(["sh", "-c", "sleep 5"], str(tmp_path), tool="sh")
    await asyncio.sleep(0.1)
    assert killed["proc"].returncode is not None


async def test_exec_runs_in_the_requested_directory(tmp_path):
    marker = tmp_path / "here.txt"
    marker.write_text("x", encoding="utf-8")
    result = await sandbox._exec(["sh", "-c", "ls here.txt"], str(tmp_path), tool="sh")
    assert result.status is CheckStatus.PASSED


# ── run_linter routing ───────────────────────────────────────────────────────


async def test_linter_skips_when_no_python_files_changed(tmp_path):
    js = tmp_path / "a.js"
    js.write_text("x", encoding="utf-8")
    result = await run_linter([str(js)], str(tmp_path))
    assert result.status is CheckStatus.SKIPPED
    assert "No Python files" in result.output


async def test_linter_skips_when_the_file_no_longer_exists(tmp_path):
    """A file the agent created then deleted must not be handed to ruff."""
    result = await run_linter([str(tmp_path / "gone.py")], str(tmp_path))
    assert result.status is CheckStatus.SKIPPED


async def test_linter_invokes_ruff_with_only_python_files(tmp_path, monkeypatch):
    captured = {}

    async def fake_exec(cmd, cwd, *, tool):
        captured["cmd"] = cmd
        captured["tool"] = tool
        return CheckResult(CheckStatus.PASSED, "")

    monkeypatch.setattr(sandbox, "_exec", fake_exec)
    py = tmp_path / "a.py"
    py.write_text("x = 1\n", encoding="utf-8")
    js = tmp_path / "b.js"
    js.write_text("x", encoding="utf-8")

    await run_linter([str(py), str(js)], str(tmp_path))
    assert captured["cmd"][0] == "ruff"
    assert str(py) in captured["cmd"]
    assert str(js) not in captured["cmd"]
    assert captured["tool"] == "ruff"


# ── run_tests routing ────────────────────────────────────────────────────────


async def test_tests_skip_when_the_project_has_none(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    result = await run_tests(str(tmp_path))
    assert result.status is CheckStatus.SKIPPED
    assert "No test files" in result.output


@pytest.mark.parametrize("filename", ["test_thing.py", "thing_test.py"])
async def test_tests_run_for_both_naming_conventions(tmp_path, monkeypatch, filename):
    called = {}

    async def fake_exec(cmd, cwd, *, tool):
        called["cmd"] = cmd
        return CheckResult(CheckStatus.PASSED, "")

    monkeypatch.setattr(sandbox, "_exec", fake_exec)
    (tmp_path / filename).write_text("def test_x(): assert True\n", encoding="utf-8")
    await run_tests(str(tmp_path))
    assert called["cmd"][0] == "pytest"


async def test_tests_are_discovered_in_nested_directories(tmp_path, monkeypatch):
    async def fake_exec(cmd, cwd, *, tool):
        return CheckResult(CheckStatus.PASSED, "")

    monkeypatch.setattr(sandbox, "_exec", fake_exec)
    nested = tmp_path / "pkg" / "tests"
    nested.mkdir(parents=True)
    (nested / "test_deep.py").write_text("def test_x(): assert True\n", encoding="utf-8")
    result = await run_tests(str(tmp_path))
    assert result.status is CheckStatus.PASSED
