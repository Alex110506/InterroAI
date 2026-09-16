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
from conftest import FakeGateway, make_response, make_tool_call

import agents.coder as coder_mod
from agents.coder import _MODEL_MAP, _REASONING_MODELS, CoderAgent
from contracts.indexing import SearchHit, SearchRequest
from core.errors import (
    FileNotFoundInProjectError,
    MissingAPIKeyError,
    PatchError,
    PathEscapeError,
)
from core.models.gateway import LONG_TIMEOUT
from core.workspace.hashing import file_hash
from core.workspace.sandbox import CheckResult, CheckStatus


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
    assert _MODEL_MAP and all(model.api_id for model in _MODEL_MAP.values())


def test_a_models_two_display_ids_differ_only_in_effort():
    """Without the effort, "low" and "high" would be the very same request."""
    low, high = _MODEL_MAP["gpt-5.4-low-effort"], _MODEL_MAP["gpt-5.4-high-effort"]

    assert low.api_id == high.api_id
    assert (low.effort, high.effort) == ("low", "high")


def test_the_chosen_effort_is_sent_to_the_planning_call(tmp_path):
    assert CoderAgent(str(tmp_path), "gpt-5.4-low-effort")._build_create_kwargs([])[
        "reasoning_effort"
    ] == "low"


def test_no_effort_is_sent_alongside_function_tools(tmp_path):
    """
    OpenAI's chat completions refuses the pair outright: "Function tools with
    reasoning_effort are not supported ... use /v1/responses". The tool rounds
    therefore go without it, and the planning call keeps it.
    """
    agent = CoderAgent(str(tmp_path), "gpt-5.4-high-effort")

    kwargs = agent._build_create_kwargs([], tools=[{"t": 1}], tool_choice="auto")

    assert "reasoning_effort" not in kwargs


def test_no_effort_is_sent_to_a_model_that_has_none(tmp_path):
    """Sending `reasoning_effort` to an ordinary model is a 400."""
    kwargs = CoderAgent(str(tmp_path), "gpt-5.4-mini")._build_create_kwargs([])
    assert "reasoning_effort" not in kwargs


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

    events = await drain(a.execute("how does this work?"))
    names = {t["function"]["name"] for t in captured["tools"]}
    assert names == {"read_file", "search_grep", "search_semantic"}
    assert events[-1]["type"] == "done"


# ── Tool loop ────────────────────────────────────────────────────────────────


async def test_tool_loop_finishes_when_the_model_stops_calling_tools(agent):
    agent._gateway = FakeGateway([make_response(content="all done")])
    assert await drain(agent._tool_loop([])) == [{"type": "impl_done", "content": "all done"}]


async def test_tool_rounds_use_the_long_timeout(agent):
    """A reasoning model's tool round can legitimately take minutes."""
    agent._gateway = gateway = FakeGateway([make_response(content="all done")])
    await drain(agent._tool_loop([]))
    assert gateway.timeouts == [LONG_TIMEOUT]


async def test_tool_loop_emits_call_and_result_events(agent):
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._gateway = FakeGateway([
        make_response(tool_calls=[call]),
        make_response(content="summary"),
    ])
    events = await drain(agent._tool_loop([]))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "impl_done"]
    assert events[0]["args"] == {"path": "main.py"}
    assert "class Greeter" in events[1]["output"]


async def test_tool_results_are_appended_to_the_conversation(agent):
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._gateway = FakeGateway([
        make_response(tool_calls=[call]),
        make_response(content="summary"),
    ])
    messages: list[dict] = []
    await drain(agent._tool_loop(messages))
    assert any(m.get("role") == "tool" and m["tool_call_id"] == "c1" for m in messages)


async def test_unparseable_tool_arguments_become_an_empty_dict(agent):
    call = make_tool_call("c1", "read_file", "{not json")
    agent._gateway = FakeGateway([
        make_response(tool_calls=[call]),
        make_response(content="done"),
    ])
    assert (await drain(agent._tool_loop([])))[0]["args"] == {}


async def test_the_tool_loop_is_bounded(agent, monkeypatch):
    """Without a cap a looping model would run until the API bill stopped it."""
    monkeypatch.setattr(coder_mod, "_MAX_TOOL_ROUNDS", 3)
    call = make_tool_call("c1", "read_file", json.dumps({"path": "main.py"}))
    agent._gateway = FakeGateway([make_response(tool_calls=[call])] * 10)

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


# ── Semantic search: locations from the index, text from the disk ───────────


class FakeIndex:
    """A `SemanticIndex` that answers every search with the same hits."""

    def __init__(self, hits=()):
        self.hits = list(hits)
        self.searches: list[SearchRequest] = []

    async def search(self, request):
        self.searches.append(request)
        return list(self.hits)


def _hit(project, path="main.py", start=4, end=6, *, indexed_hash=None):
    """A hit for lines 4-6 of the sample `main.py` — the `Greeter` class."""
    return SearchHit(
        file_path=path,
        start_line=start,
        end_line=end,
        file_hash=file_hash(project / path) if indexed_hash is None else indexed_hash,
        score=0.9,
    )


def _searching_agent(project, hits):
    index = FakeIndex(hits)
    return CoderAgent(str(project), "gpt-5.4-mini", index=index), index


async def test_search_reads_the_code_from_the_working_tree(tmp_project):
    """The index holds no text; what the model sees must come from disk."""
    agent, _ = _searching_agent(tmp_project, [_hit(tmp_project)])
    out = await agent._search_semantic("greeting")

    assert "class Greeter" in out
    assert "main.py" in out and "lines 4–6" in out
    assert "stale" not in out


async def test_search_asks_the_index_about_this_project(tmp_project):
    agent, index = _searching_agent(tmp_project, [])
    await agent._search_semantic("greeting", n=3)

    [request] = index.searches
    assert request.project_id == str(tmp_project.resolve())
    assert request.query == "greeting"
    assert request.n == 3


@pytest.mark.parametrize(("asked", "sent"), [(0, 1), (50, 10)])
async def test_the_result_count_is_clamped(tmp_project, asked, sent):
    agent, index = _searching_agent(tmp_project, [])
    await agent._search_semantic("x", n=asked)
    assert index.searches[0].n == sent


async def test_a_hit_from_an_edited_file_is_labelled_stale(tmp_project):
    """
    The lines are usually close, so the hit is still shown — but the model is
    told they may have moved, and to read the file before trusting them.
    """
    agent, _ = _searching_agent(tmp_project, [_hit(tmp_project, indexed_hash="an-older-hash")])
    out = await agent._search_semantic("greeting")

    assert "stale" in out
    assert "class Greeter" in out


async def test_a_hit_for_a_deleted_file_is_dropped(tmp_project):
    agent, _ = _searching_agent(tmp_project, [_hit(tmp_project, "gone.py", indexed_hash="h")])
    assert await agent._search_semantic("x") == "No semantically similar code found."


async def test_a_hit_past_the_end_of_a_shrunken_file_is_dropped(tmp_project):
    agent, _ = _searching_agent(tmp_project, [_hit(tmp_project, start=400, end=420)])
    assert await agent._search_semantic("x") == "No semantically similar code found."


async def test_a_hit_outside_the_project_is_never_read(tmp_project, tmp_path):
    """An index is not trusted to name the files the agent may open."""
    (tmp_path / "outside.py").write_text("PRIVATE = 'outside the project'\n", encoding="utf-8")
    hit = SearchHit(
        file_path="../outside.py", start_line=1, end_line=1, file_hash="h", score=0.9
    )
    agent, _ = _searching_agent(tmp_project, [hit])

    out = await agent._search_semantic("x")
    assert "PRIVATE" not in out


async def test_the_search_tool_is_dispatched_through_the_index(tmp_project):
    agent, index = _searching_agent(tmp_project, [_hit(tmp_project)])
    out = await agent._run_tool("search_semantic", {"query": "greeting"})

    assert index.searches, "the tool must go through the injected index"
    assert "class Greeter" in out


# ── Conversation context ─────────────────────────────────────────────────────

_PRIOR = [
    {"role": "user", "content": "what is this project about"},
    {"role": "assistant", "content": "A social chat and calling app."},
]


async def _captured_messages(agent, monkeypatch, prompt="do the thing"):
    """Run one tool loop and return the messages the model was shown."""
    agent._gateway = gateway = FakeGateway([make_response(content="done")])
    monkeypatch.setattr(agent, "_build_knowledge_tree", _fake_tree)
    monkeypatch.setattr(agent, "_validate", _no_validation)
    [e async for e in agent.execute(prompt)]
    return gateway.requests[0]["messages"] if gateway.requests else []


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


async def test_planning_sees_the_earlier_turns(tmp_project):
    gateway = FakeGateway([make_response(content="1. do it")])
    agent = CoderAgent(
        str(tmp_project), "gpt-5.5-high-effort", history=_PRIOR, gateway=gateway
    )
    [e async for e in agent._plan("now add tests", "<repo_map/>")]

    contents = [m["content"] for m in gateway.requests[0]["messages"]]
    assert "A social chat and calling app." in contents


async def test_no_history_leaves_the_prompt_unchanged(tmp_project, monkeypatch):
    """The default path must not grow an empty turn."""
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", intent="answer")
    messages = await _captured_messages(agent, monkeypatch, "what is this?")
    assert len(messages) == 2, "system + the request, nothing else"
