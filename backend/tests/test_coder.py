"""
Coder Agent.

Two areas carry the most risk and get the most attention here:

  * `_resolve` — the sandbox boundary. It is the only thing standing between a
    model-supplied path and the user's filesystem.
  * `_run_tool` — the expected/unexpected split. A `ToolError` is handed to the
    model to fix; a real defect must be logged with its traceback instead of
    being stringified into the transcript as if the model caused it.
"""
from __future__ import annotations

import json

import pytest
from conftest import make_response, make_tool_call

import agents.coder as coder_mod
from agents.coder import _MODEL_MAP, _REASONING_MODELS, CoderAgent, _fmt_chunks
from core.errors import (
    FileNotFoundInProjectError,
    MissingAPIKeyError,
    PatchError,
    PathEscapeError,
)
from core.sandbox import CheckResult, CheckStatus


@pytest.fixture
def agent(tmp_project):
    return CoderAgent(project_path=str(tmp_project), model="gpt-5.4-mini")


async def drain(generator):
    return [event async for event in generator]


def _async(value):
    """Wrap a plain value in an awaitable, for monkeypatched async functions."""
    async def _inner():
        return value
    return _inner()


# ── Model resolution ─────────────────────────────────────────────────────────


def test_every_display_id_maps_to_an_api_id():
    assert _MODEL_MAP and all(isinstance(v, str) and v for v in _MODEL_MAP.values())


def test_display_id_is_translated(tmp_path):
    assert CoderAgent(str(tmp_path), "gpt-5.5-high-effort")._api_model == "gpt-5.5"


def test_unknown_id_passes_through_unchanged(tmp_path):
    """Keeps a raw OpenAI model name usable without a registry entry."""
    assert CoderAgent(str(tmp_path), "gpt-4o-mini")._api_model == "gpt-4o-mini"


@pytest.mark.parametrize("display", ["gpt-5.4-low-effort", "gpt-5.5-high-effort"])
def test_reasoning_models_are_detected(tmp_path, display):
    assert CoderAgent(str(tmp_path), display)._is_reasoning is True


def test_non_reasoning_model_is_detected(tmp_path):
    assert CoderAgent(str(tmp_path), "gpt-5.4-mini")._is_reasoning is False


def test_temperature_is_omitted_for_reasoning_models(tmp_path):
    """Reasoning models reject `temperature` outright — sending it is a 400."""
    a = CoderAgent(str(tmp_path), "gpt-5.5-high-effort")
    assert "temperature" not in a._build_create_kwargs([], temperature=0.2)


def test_temperature_is_sent_for_ordinary_models(tmp_path):
    a = CoderAgent(str(tmp_path), "gpt-5.4-mini")
    assert a._build_create_kwargs([], temperature=0.2)["temperature"] == 0.2


def test_extra_kwargs_are_forwarded(tmp_path):
    a = CoderAgent(str(tmp_path), "gpt-5.4-mini")
    kwargs = a._build_create_kwargs([], tools=[{"t": 1}], tool_choice="auto")
    assert kwargs["tools"] == [{"t": 1}] and kwargs["tool_choice"] == "auto"


def test_reasoning_model_set_is_not_empty():
    assert _REASONING_MODELS


# ── Sandbox boundary ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hostile",
    [
        "../outside.py",
        "../../etc/passwd",
        "utils/../../escape.py",
        "/etc/passwd",
    ],
)
def test_paths_outside_the_project_are_rejected(agent, hostile):
    with pytest.raises(PathEscapeError):
        agent._resolve(hostile)


@pytest.mark.parametrize("ok", ["main.py", "utils/helpers.py", "./main.py", "utils/../main.py"])
def test_paths_inside_the_project_resolve(agent, ok, tmp_project):
    assert agent._resolve(ok).is_relative_to(tmp_project.resolve())


def test_the_project_root_itself_resolves(agent, tmp_project):
    assert agent._resolve(".") == tmp_project.resolve()


def test_a_symlink_pointing_outside_is_rejected(agent, tmp_project, tmp_path):
    """`is_relative_to` on a resolved path is what makes this work."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_project / "link.txt").symlink_to(outside)
    with pytest.raises(PathEscapeError):
        agent._resolve("link.txt")


def test_a_sibling_prefix_directory_is_rejected(tmp_path):
    """`startswith` would wrongly accept `/x/proj-evil` as inside `/x/proj`."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj-evil").mkdir()
    a = CoderAgent(str(tmp_path / "proj"), "gpt-5.4-mini")
    with pytest.raises(PathEscapeError):
        a._resolve("../proj-evil/x.py")


# ── read_file ────────────────────────────────────────────────────────────────


def test_read_returns_file_contents(agent):
    assert "class Greeter" in agent._read_file("main.py", None, None)


def test_reading_a_missing_file_raises_a_tool_error(agent):
    with pytest.raises(FileNotFoundInProjectError):
        agent._read_file("nope.py", None, None)


def test_read_honours_a_line_range(agent, tmp_project):
    (tmp_project / "nums.py").write_text(
        "".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8"
    )
    assert agent._read_file("nums.py", 3, 5) == "line3\nline4\nline5\n"


def test_read_from_a_start_line_to_the_end(agent, tmp_project):
    (tmp_project / "nums.py").write_text("a\nb\nc\n", encoding="utf-8")
    assert agent._read_file("nums.py", 2, None) == "b\nc\n"


def test_large_files_are_truncated_with_a_hint(agent, tmp_project):
    (tmp_project / "huge.py").write_text("x" * 20_000, encoding="utf-8")
    out = agent._read_file("huge.py", None, None)
    assert len(out) < 20_000
    assert "truncated" in out


# ── write_file / patch_file ──────────────────────────────────────────────────


def test_write_creates_the_file_and_parent_directories(agent, tmp_project):
    agent._write_file("deep/nested/new.py", "x = 1\n")
    assert (tmp_project / "deep" / "nested" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_records_the_file_for_validation(agent, tmp_project):
    agent._write_file("new.py", "x = 1\n")
    assert str((tmp_project / "new.py").resolve()) in agent._modified


def test_write_outside_the_project_is_refused(agent, tmp_path):
    with pytest.raises(PathEscapeError):
        agent._write_file("../escaped.py", "x = 1\n")
    assert not (tmp_path / "escaped.py").exists()


def test_patch_records_the_file_for_validation(agent, tmp_project):
    agent._patch_file("utils/helpers.py", "text.lower()", "text.upper()")
    assert str((tmp_project / "utils" / "helpers.py").resolve()) in agent._modified


def test_a_failed_patch_does_not_record_the_file(agent):
    """Only genuinely-changed files should be handed to the linter."""
    with pytest.raises(PatchError):
        agent._patch_file("utils/helpers.py", "not present", "x")
    assert agent._modified == set()


# ── search_grep ──────────────────────────────────────────────────────────────


def test_grep_finds_a_literal_and_reports_file_and_line(agent):
    assert "utils/helpers.py:1" in agent._search_grep("slugify", None)


def test_grep_supports_regex(agent):
    assert "main.py" in agent._search_grep(r"class\s+\w+", None)


def test_an_invalid_regex_is_treated_as_a_literal(agent, tmp_project):
    """A model writing `foo(` should get results, not a crash."""
    (tmp_project / "lit.py").write_text("value = f(\n", encoding="utf-8")
    assert "lit.py" in agent._search_grep("f(", None)


def test_grep_can_be_limited_by_glob(agent):
    out = agent._search_grep("e", "*.md")
    assert "README.md" in out
    assert "main.py" not in out


def test_grep_reports_no_matches_clearly(agent):
    assert "No matches" in agent._search_grep("zzz-nonexistent-zzz", None)


def test_grep_skips_vendor_directories(agent):
    assert "node_modules" not in agent._search_grep("function", None)


def test_grep_caps_its_output(agent, tmp_project):
    (tmp_project / "many.py").write_text("hit\n" * 500, encoding="utf-8")
    out = agent._search_grep("hit", None)
    assert "capped at 50" in out
    assert len(out.splitlines()) <= 51


# ── _run_tool: the expected / unexpected split ───────────────────────────────


async def test_tool_errors_are_returned_to_the_model(agent):
    """The model can fix this itself, so it gets the message verbatim."""
    result = await agent._run_tool("read_file", {"path": "missing.py"})
    assert result.startswith("Error:")
    assert "File not found" in result


async def test_path_escapes_are_reported_without_crashing_the_run(agent):
    result = await agent._run_tool("read_file", {"path": "../../etc/passwd"})
    assert "escapes the project directory" in result


async def test_an_unknown_tool_name_is_reported(agent):
    assert "Unknown tool" in await agent._run_tool("delete_everything", {})


async def test_terminal_errors_are_re_raised_not_returned(agent, monkeypatch):
    """
    A missing key must stop the run. Returning it to the model would burn all
    20 tool rounds "retrying" something no model can fix.
    """
    async def boom(query, n=5):
        raise MissingAPIKeyError()

    monkeypatch.setattr(agent, "_search_semantic", boom)
    with pytest.raises(MissingAPIKeyError):
        await agent._run_tool("search_semantic", {"query": "x"})


async def test_unexpected_failures_are_logged_with_a_traceback(agent, monkeypatch, caplog):
    """The anti-pattern being guarded: a real defect silently stringified."""
    def explode(*args, **kwargs):
        raise RuntimeError("genuine defect")

    monkeypatch.setattr(agent, "_read_file", explode)
    with caplog.at_level("ERROR", logger="agents.coder"):
        result = await agent._run_tool("read_file", {"path": "main.py"})

    assert "Internal error" in result
    assert "genuine defect" not in result, "internals must not leak to the model"
    assert any(r.exc_info for r in caplog.records), "the traceback must be preserved"


async def test_malformed_tool_arguments_do_not_crash(agent):
    assert await agent._run_tool("read_file", {})


# ── Validation phase ─────────────────────────────────────────────────────────


async def test_validation_is_skipped_when_nothing_changed(agent):
    events = await drain(agent._validate([]))
    assert len(events) == 1
    assert events[0]["status"] == "skipped"
    assert events[0]["passed"] is True


async def test_a_clean_run_emits_lint_and_test_results(agent, monkeypatch):
    monkeypatch.setattr(coder_mod, "run_linter",
                        lambda *a, **k: _async(CheckResult(CheckStatus.PASSED, "")))
    monkeypatch.setattr(coder_mod, "run_tests",
                        lambda *a, **k: _async(CheckResult(CheckStatus.PASSED, "")))
    agent._modified.add("/x.py")

    events = await drain(agent._validate([]))
    assert [e["phase"] for e in events] == ["lint", "test"]
    assert all(e["passed"] for e in events)


async def test_skipped_checks_are_reported_as_skipped_not_passed(agent, monkeypatch):
    """The event must let the UI say "skipped" rather than claim success."""
    monkeypatch.setattr(coder_mod, "run_linter",
                        lambda *a, **k: _async(CheckResult(CheckStatus.SKIPPED, "ruff missing")))
    monkeypatch.setattr(coder_mod, "run_tests",
                        lambda *a, **k: _async(CheckResult(CheckStatus.SKIPPED, "no tests")))
    agent._modified.add("/x.py")

    events = await drain(agent._validate([]))
    assert all(e["status"] == "skipped" for e in events)
    assert all(e["passed"] for e in events)


async def test_a_failure_triggers_a_correction_round(agent, monkeypatch):
    monkeypatch.setattr(coder_mod, "run_linter",
                        lambda *a, **k: _async(CheckResult(CheckStatus.FAILED, "E501 too long")))
    monkeypatch.setattr(coder_mod, "run_tests",
                        lambda *a, **k: _async(CheckResult(CheckStatus.PASSED, "")))

    async def fake_loop(messages, tools=None):
        yield {"type": "impl_done", "content": "fixed"}

    monkeypatch.setattr(agent, "_tool_loop", fake_loop)
    agent._modified.add("/x.py")

    events = await drain(agent._validate([]))
    corrections = [e for e in events if e["type"] == "correction"]
    assert corrections
    assert "E501 too long" in corrections[0]["errors"]


async def test_corrections_are_bounded(agent, monkeypatch):
    """A model that cannot fix the error must not loop forever."""
    monkeypatch.setattr(coder_mod, "run_linter",
                        lambda *a, **k: _async(CheckResult(CheckStatus.FAILED, "still broken")))
    monkeypatch.setattr(coder_mod, "run_tests",
                        lambda *a, **k: _async(CheckResult(CheckStatus.PASSED, "")))

    async def fake_loop(messages, tools=None):
        yield {"type": "impl_done", "content": "attempted"}

    monkeypatch.setattr(agent, "_tool_loop", fake_loop)
    agent._modified.add("/x.py")

    events = await drain(agent._validate([]))
    assert len([e for e in events if e["type"] == "correction"]) < coder_mod._MAX_CORRECTIONS


# ── execute(): top-level guards ──────────────────────────────────────────────


async def test_a_missing_project_directory_is_an_error_event(tmp_path):
    a = CoderAgent(str(tmp_path / "nope"), "gpt-5.4-mini")
    events = await drain(a.execute("do something"))
    assert events[0]["type"] == "error"
    assert "not found" in events[0]["message"]


async def test_a_missing_api_key_is_surfaced_verbatim(tmp_project, without_api_key):
    """The user must be told to add a key, not shown a generic failure."""
    a = CoderAgent(str(tmp_project), "gpt-5.4-mini")
    events = await drain(a.execute("do something"))
    assert events[0]["type"] == "error"
    assert "API key" in events[0]["message"]


async def test_the_answer_intent_never_exposes_write_tools(tmp_project, monkeypatch):
    """A question must not be able to mutate the repository."""
    captured = {}

    async def fake_loop(messages, tools=None):
        captured["tools"] = tools
        yield {"type": "impl_done", "content": "an answer"}

    a = CoderAgent(str(tmp_project), "gpt-5.4-mini", intent="answer")
    monkeypatch.setattr(a, "_tool_loop", fake_loop)
    monkeypatch.setattr(coder_mod, "get_client", lambda timeout: object())

    events = await drain(a.execute("how does this work?"))
    names = {t["function"]["name"] for t in captured["tools"]}
    assert names == {"read_file", "search_grep", "search_semantic"}
    assert events[-1]["type"] == "done"


# ── Tool loop ────────────────────────────────────────────────────────────────


class _FakeClient:
    """Minimal stand-in for AsyncOpenAI inside the tool loop."""

    def __init__(self, responses):
        self._responses = list(responses)
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                return outer._responses.pop(0)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


async def test_tool_loop_finishes_when_the_model_stops_calling_tools(agent):
    agent._client = _FakeClient([make_response(content="all done")])
    assert await drain(agent._tool_loop([])) == [{"type": "impl_done", "content": "all done"}]


async def test_tool_loop_emits_call_and_result_events(agent):
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._client = _FakeClient([
        make_response(tool_calls=[call]),
        make_response(content="summary"),
    ])
    events = await drain(agent._tool_loop([]))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "impl_done"]
    assert events[0]["args"] == {"path": "main.py"}
    assert "class Greeter" in events[1]["output"]


async def test_tool_results_are_appended_to_the_conversation(agent):
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._client = _FakeClient([
        make_response(tool_calls=[call]),
        make_response(content="summary"),
    ])
    messages: list[dict] = []
    await drain(agent._tool_loop(messages))
    assert any(m.get("role") == "tool" and m["tool_call_id"] == "c1" for m in messages)


async def test_unparseable_tool_arguments_become_an_empty_dict(agent):
    call = make_tool_call("c1", "read_file", "{not json")
    agent._client = _FakeClient([
        make_response(tool_calls=[call]),
        make_response(content="done"),
    ])
    assert (await drain(agent._tool_loop([])))[0]["args"] == {}


async def test_the_tool_loop_is_bounded(agent, monkeypatch):
    """Without a cap a looping model would run until the API bill stopped it."""
    monkeypatch.setattr(coder_mod, "_MAX_TOOL_ROUNDS", 3)
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._client = _FakeClient([make_response(tool_calls=[call])] * 10)

    events = await drain(agent._tool_loop([]))
    assert len([e for e in events if e["type"] == "tool_call"]) == 3
    assert "limit reached" in events[-1]["content"]


# ── Knowledge tree & formatting ──────────────────────────────────────────────


async def test_knowledge_tree_contains_the_repo_map(agent):
    tree = await agent._build_knowledge_tree()
    assert "<repo_map>" in tree
    assert "class Greeter" in tree


async def test_knowledge_tree_survives_an_empty_project(tmp_path):
    a = CoderAgent(str(tmp_path), "gpt-5.4-mini")
    assert "(empty)" in await a._build_knowledge_tree()


async def test_the_repo_map_is_not_rebuilt_for_an_unchanged_project(agent, monkeypatch):
    """
    The agent needs the map on every request, and parsing every source file
    again to produce an identical map was the most expensive thing it did
    before the first token.
    """
    walks = 0
    real_build = coder_mod.build_repo_map

    def counted(path):
        nonlocal walks
        walks += 1
        return real_build(path)

    monkeypatch.setattr(coder_mod, "build_repo_map", counted)

    first = await agent._build_knowledge_tree()
    second = await agent._build_knowledge_tree()

    assert walks == 1, "the second request must be served from the cache"
    assert first == second


async def test_a_changed_file_rebuilds_the_repo_map(agent, tmp_project, monkeypatch):
    walks = 0
    real_build = coder_mod.build_repo_map

    def counted(path):
        nonlocal walks
        walks += 1
        return real_build(path)

    monkeypatch.setattr(coder_mod, "build_repo_map", counted)
    await agent._build_knowledge_tree()

    (tmp_project / "main.py").write_text("class Rewritten:\n    pass\n", encoding="utf-8")
    tree = await agent._build_knowledge_tree()

    assert walks == 2, "a stale map must not be served"
    assert "class Rewritten" in tree


async def test_the_knowledge_tree_still_builds_without_redis(agent, fake_redis):
    """The cache is an accelerator; the agent cannot depend on it."""
    fake_redis.broken = True
    assert "class Greeter" in await agent._build_knowledge_tree()


def test_chunk_formatting_includes_file_and_line_range():
    out = _fmt_chunks([{"file_path": "a.py", "start_line": 1, "end_line": 9, "content": "code"}])
    assert "a.py" in out and "1" in out and "9" in out


def test_chunk_formatting_handles_no_results():
    assert "No relevant code" in _fmt_chunks([])


# ── Conversation context ─────────────────────────────────────────────────────

_PRIOR = [
    {"role": "user", "content": "what is this project about"},
    {"role": "assistant", "content": "A social chat and calling app."},
]


async def _captured_messages(agent, monkeypatch, prompt="do the thing"):
    """Run one tool loop and return the messages the model was shown."""
    captured = {}

    async def capture(client, **kwargs):
        # Snapshot: `_tool_loop` appends the reply to the same list afterwards.
        captured["messages"] = list(kwargs["messages"])
        return make_response(content="done")

    monkeypatch.setattr(coder_mod, "get_client", lambda timeout: object())
    monkeypatch.setattr(coder_mod, "chat_completion", capture)
    monkeypatch.setattr(agent, "_build_knowledge_tree", _fake_tree)
    monkeypatch.setattr(agent, "_validate", _no_validation)
    [e async for e in agent.execute(prompt)]
    return captured.get("messages", [])


async def _fake_tree():
    return "<repo_map/>"


async def _no_validation(messages):
    return
    yield  # pragma: no cover


async def test_a_question_sees_the_earlier_turns(tmp_project, monkeypatch):
    """
    The reported bug: a follow-up reached the Q&A agent with no conversation,
    so "sure" had nothing to agree to.
    """
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", intent="answer", history=_PRIOR)
    messages = await _captured_messages(agent, monkeypatch, "sure")

    contents = [m["content"] for m in messages]
    assert "A social chat and calling app." in contents
    assert contents[0].startswith("You are"), "the system prompt must stay first"
    assert "sure" in contents[-1], "the new request must stay last"


async def test_an_implementation_sees_the_earlier_turns(tmp_project, monkeypatch):
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", history=_PRIOR)

    async def fake_plan(prompt, tree):
        yield {"type": "plan", "content": "the plan"}

    monkeypatch.setattr(agent, "_plan", fake_plan)
    messages = await _captured_messages(agent, monkeypatch, "now add tests")

    contents = [m["content"] for m in messages]
    assert "A social chat and calling app." in contents
    assert "now add tests" in contents[-1]


async def test_planning_sees_the_earlier_turns(tmp_project, monkeypatch):
    agent = CoderAgent(str(tmp_project), "gpt-5.5-high-effort", history=_PRIOR)
    captured = {}

    async def capture(client, **kwargs):
        captured["messages"] = list(kwargs["messages"])
        return make_response(content="1. do it")

    monkeypatch.setattr(coder_mod, "chat_completion", capture)
    [e async for e in agent._plan("now add tests", "<repo_map/>")]

    contents = [m["content"] for m in captured["messages"]]
    assert "A social chat and calling app." in contents


async def test_no_history_leaves_the_prompt_unchanged(tmp_project, monkeypatch):
    """The default path must not grow an empty turn."""
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", intent="answer")
    messages = await _captured_messages(agent, monkeypatch, "what is this?")
    assert len(messages) == 2, "system + the request, nothing else"
