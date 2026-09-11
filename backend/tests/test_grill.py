"""
Grill Agent — the clarification loop and the context it builds.

The behaviour worth protecting is the turn budget (an agent that asks forever
is unusable) and the refusal to disguise a missing API key as a RAG problem.
"""
from __future__ import annotations

import json

import pytest
from conftest import make_response

import agents.grill as grill_mod
from agents.grill import _MAX_TURNS, GrillAgent, _fmt_chunks, _fmt_git, _fmt_tree
from core.errors import MissingAPIKeyError


def _ready(prompt="a full spec"):
    return json.dumps({"is_prompt_ready": True, "refined_prompt": prompt})


def _question(text="which file?"):
    return json.dumps({"is_prompt_ready": False, "question": text})


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner


@pytest.fixture
def agent(monkeypatch):
    """A GrillAgent with its context builder stubbed out."""
    a = GrillAgent(project_path="/proj", project_index={})

    async def no_context(query):
        return "CONTEXT"

    monkeypatch.setattr(a, "_build_context", no_context)
    return a


def _queue(monkeypatch, *payloads):
    """Make `_call` see the given JSON payloads, in order."""
    responses = [make_response(content=p) for p in payloads]

    async def fake_completion(client, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(grill_mod, "get_client", lambda timeout: object())
    monkeypatch.setattr(grill_mod, "chat_completion", fake_completion)


# ── Tree formatting ──────────────────────────────────────────────────────────


def test_empty_tree_renders_as_empty_string():
    assert _fmt_tree({}) == ""


def test_a_file_node_renders_its_name():
    assert _fmt_tree({"kind": "file", "name": "main.py"}).strip() == "main.py"


def test_a_directory_renders_with_a_trailing_slash():
    assert _fmt_tree({"kind": "dir", "name": "src", "children": []}).strip() == "src/"


def test_children_are_indented_under_their_parent():
    out = _fmt_tree({
        "kind": "dir", "name": "src",
        "children": [{"kind": "file", "name": "app.py"}],
    })
    assert "src/" in out
    assert "  app.py" in out


def test_tree_depth_is_capped():
    """An unbounded tree would crowd out the actual code context."""
    node = {"kind": "file", "name": "deep.py"}
    for level in range(12):
        node = {"kind": "dir", "name": f"d{level}", "children": [node]}
    assert "deep.py" not in _fmt_tree(node)


def test_a_directory_without_children_does_not_crash():
    assert _fmt_tree({"kind": "dir", "name": "empty"}).strip() == "empty/"


# ── Git formatting ───────────────────────────────────────────────────────────


def test_absent_git_context_is_stated_plainly():
    assert _fmt_git({}) == "Not a git repository."
    assert _fmt_git({"is_git_repo": False}) == "Not a git repository."


def test_a_clean_repository_is_reported_as_clean():
    assert _fmt_git({"is_git_repo": True}) == "Clean working tree."


def test_branch_modified_files_and_commits_are_rendered():
    out = _fmt_git({
        "is_git_repo": True,
        "branch": "feature/login",
        "modified_files": ["api/auth.py"],
        "recent_commits": ["abc123 first"],
    })
    assert "feature/login" in out
    assert "api/auth.py" in out
    assert "abc123 first" in out


def test_chunk_formatting_reports_an_empty_store():
    assert "No relevant code" in _fmt_chunks([])


# ── Context assembly ─────────────────────────────────────────────────────────


async def test_context_includes_all_three_layers(monkeypatch):
    a = GrillAgent("/proj", {"file_tree": {"kind": "dir", "name": "src", "children": []}})
    monkeypatch.setattr(grill_mod, "embed_texts", _async_return([[0.1]]))
    monkeypatch.setattr(grill_mod, "search_chunks", lambda *a, **k: [])

    context = await a._build_context("add login")
    assert "PROJECT STRUCTURE:" in context
    assert "GIT CONTEXT:" in context
    assert "RELEVANT CODE:" in context


async def test_a_broken_vector_store_degrades_gracefully(monkeypatch, caplog):
    """RAG is optional here — the tree and git context still make a good question."""
    a = GrillAgent("/proj", {})

    async def boom(texts):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(grill_mod, "embed_texts", boom)
    with caplog.at_level("WARNING", logger="agents.grill"):
        context = await a._build_context("q")

    assert "Vector search unavailable" in context
    assert any("Vector search failed" in r.message for r in caplog.records), (
        "a persistently broken store must be visible in the logs"
    )


async def test_a_missing_key_is_not_disguised_as_a_rag_failure(monkeypatch):
    """
    Reporting "vector search unavailable" would send the user hunting through
    their index while the real cause is an unset API key.
    """
    a = GrillAgent("/proj", {})

    async def boom(texts):
        raise MissingAPIKeyError()

    monkeypatch.setattr(grill_mod, "embed_texts", boom)
    with pytest.raises(MissingAPIKeyError):
        await a._build_context("q")


# ── Session flow ─────────────────────────────────────────────────────────────


async def test_start_returns_the_first_question(agent, monkeypatch):
    _queue(monkeypatch, _question("which endpoint?"))
    result = await agent.start("add auth")
    assert result["is_prompt_ready"] is False
    assert result["question"] == "which endpoint?"


async def test_start_can_finish_immediately(agent, monkeypatch):
    """An already-specific request should not be interrogated at all."""
    _queue(monkeypatch, _ready("rename X to Y in a.py"))
    result = await agent.start("rename X to Y in a.py")
    assert result["is_prompt_ready"] is True
    assert result["refined_prompt"] == "rename X to Y in a.py"


async def test_the_initial_prompt_is_recorded_in_the_history(agent, monkeypatch):
    _queue(monkeypatch, _question())
    await agent.start("add auth")
    assert agent._history[0]["role"] == "user"
    assert "add auth" in agent._history[0]["content"]


async def test_questions_increment_the_turn_counter(agent, monkeypatch):
    _queue(monkeypatch, _question(), _question())
    await agent.start("vague")
    assert agent._turns == 1
    await agent.answer("some detail")
    assert agent._turns == 2


async def test_finishing_does_not_increment_the_turn_counter(agent, monkeypatch):
    _queue(monkeypatch, _ready())
    await agent.start("clear request")
    assert agent._turns == 0


async def test_answers_are_appended_to_the_history(agent, monkeypatch):
    _queue(monkeypatch, _question(), _question())
    await agent.start("vague")
    await agent.answer("my answer")
    assert {"role": "user", "content": "my answer"} in agent._history


async def test_the_question_budget_forces_a_conclusion(agent, monkeypatch):
    """
    Once the cap is hit the next answer must produce a refined prompt rather
    than another question, however the model would prefer to continue.
    """
    _queue(monkeypatch, *([_question()] * _MAX_TURNS), _ready("consolidated"))
    await agent.start("vague")
    for _ in range(_MAX_TURNS - 1):
        await agent.answer("more detail")

    assert agent._turns == _MAX_TURNS
    result = await agent.answer("final detail")
    assert result["is_prompt_ready"] is True
    assert result["refined_prompt"] == "consolidated"


async def test_force_ready_consolidates_early(agent, monkeypatch):
    """Backs the UI's "skip the questions" button."""
    _queue(monkeypatch, _question(), _ready("best effort spec"))
    await agent.start("vague")
    result = await agent._force_ready()
    assert result["is_prompt_ready"] is True
    assert result["refined_prompt"] == "best effort spec"


async def test_json_mode_is_requested(agent, monkeypatch):
    """The caller parses the reply with `json.loads` and has no fallback."""
    captured = {}

    async def capture(client, **kwargs):
        captured.update(kwargs)
        return make_response(content=_ready())

    monkeypatch.setattr(grill_mod, "get_client", lambda timeout: object())
    monkeypatch.setattr(grill_mod, "chat_completion", capture)
    await agent.start("x")
    assert captured["response_format"] == {"type": "json_object"}


async def test_a_missing_key_stops_the_session(agent, monkeypatch):
    def boom(timeout):
        raise MissingAPIKeyError()

    monkeypatch.setattr(grill_mod, "get_client", boom)
    with pytest.raises(MissingAPIKeyError):
        await agent.start("anything")
