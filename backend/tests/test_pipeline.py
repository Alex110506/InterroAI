"""
End-to-end paths through the two WebSocket pipelines.

The unit tests elsewhere cover each component in isolation; these drive the
actual sockets so the routing between them — which intent leads where, and what
the client receives in what order — is pinned down too.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import make_response
from fastapi.testclient import TestClient

import agents.coder as coder_mod
import agents.session as session
import core.project_index as project_index
from agents.coder import CoderAgent
from core.embeddings import EmbeddedBatch
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def captured_supervisor(monkeypatch):
    """Replace the supervisor so no agent or network work actually happens."""
    calls: list[dict] = []

    async def fake_stream(prompt, project_path, model, intent="implement", history=None):
        calls.append(
            {"prompt": prompt, "project_path": project_path, "model": model, "intent": intent}
        )
        yield {"type": "done", "summary": "stubbed"}

    monkeypatch.setattr(session.supervisor, "stream", fake_stream)
    return calls


def _fix_intent(monkeypatch, action):
    async def fake_classify(message, index, history=None):
        return action

    monkeypatch.setattr(session, "classify_intent", fake_classify)


def _start(**overrides):
    payload = {
        "type": "start",
        "project_path": "/tmp/proj",
        "project_index": {},
        "message": "do the thing",
        "model": "gpt-5.4-mini",
    }
    payload.update(overrides)
    return payload


# ── Chat pipeline: routing by intent ─────────────────────────────────────────


def test_an_answer_goes_straight_to_the_qa_agent(client, monkeypatch, captured_supervisor):
    _fix_intent(monkeypatch, "answer")
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(message="what does this repo do?"))
        ready = ws.receive_json()
        done = ws.receive_json()

    assert ready["type"] == "ready"
    assert done["type"] == "done"
    assert captured_supervisor[0]["intent"] == "answer"
    assert captured_supervisor[0]["prompt"] == "what does this repo do?"


def test_a_clear_request_goes_straight_to_the_coder(client, monkeypatch, captured_supervisor):
    _fix_intent(monkeypatch, "implement")
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(message="rename X to Y in a.py"))
        ready = ws.receive_json()
        ws.receive_json()

    assert ready["type"] == "ready"
    assert captured_supervisor[0]["intent"] == "implement"
    assert captured_supervisor[0]["prompt"] == "rename X to Y in a.py"


def test_the_selected_model_reaches_the_supervisor(client, monkeypatch, captured_supervisor):
    """Auto-routing is gone, so whatever the user picked must be used verbatim."""
    _fix_intent(monkeypatch, "implement")
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(model="gpt-5.5-high-effort"))
        ws.receive_json()
        ws.receive_json()

    assert captured_supervisor[0]["model"] == "gpt-5.5-high-effort"


def test_omitting_the_model_falls_back_to_the_default(client, monkeypatch, captured_supervisor):
    _fix_intent(monkeypatch, "implement")
    payload = _start()
    del payload["model"]
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(payload)
        ws.receive_json()
        ws.receive_json()

    assert captured_supervisor[0]["model"] == session._DEFAULT_MODEL


def test_the_project_path_reaches_the_supervisor(client, monkeypatch, captured_supervisor):
    _fix_intent(monkeypatch, "implement")
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(project_path="/some/where"))
        ws.receive_json()
        ws.receive_json()

    assert captured_supervisor[0]["project_path"] == "/some/where"


# ── Indexing pipeline ────────────────────────────────────────────────────────


@pytest.fixture
def stub_embedding(monkeypatch):
    stored = {}

    async def fake_batches(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(len(texts))),
            vectors=[[0.1] * 4 for _ in texts],
        )

    def fake_store(path, chunks, embeddings):
        stored["path"] = path
        # Called once per batch now, so accumulate rather than overwrite.
        stored.setdefault("chunks", []).extend(chunks)

    monkeypatch.setattr(project_index, "embed_batches", fake_batches)
    monkeypatch.setattr(project_index, "store_chunks", fake_store)
    return stored


def _drain_steps(ws):
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["step"] in {"done", "error"}:
            return events


def test_indexing_emits_every_step_in_order(client, tmp_project, stub_embedding):
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_project)})
        events = _drain_steps(ws)

    steps = [e["step"] for e in events]
    assert steps[0] == "A"
    assert "B" in steps and "C" in steps and "D" in steps
    assert steps[-1] == "done"


def test_indexing_reports_the_file_and_chunk_counts(client, tmp_project, stub_embedding):
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_project)})
        events = _drain_steps(ws)

    a_done = next(e for e in events if e["step"] == "A" and e.get("status") == "done")
    b_done = next(e for e in events if e["step"] == "B" and e.get("status") == "done")
    assert a_done["files"] > 0
    assert b_done["chunks"] > 0


def test_indexing_persists_the_chunks(client, tmp_project, stub_embedding):
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_project)})
        _drain_steps(ws)

    assert stub_embedding["path"] == str(tmp_project.resolve())
    assert stub_embedding["chunks"]


def test_indexing_reports_embedding_progress(client, tmp_project, stub_embedding):
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_project)})
        events = _drain_steps(ws)

    assert any(e["step"] == "C" and e.get("status") == "progress" for e in events)


def test_indexing_rejects_a_non_directory(client, tmp_path):
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_path / "nope")})
        event = ws.receive_json()

    assert event["step"] == "error"
    assert "Not a directory" in event["message"]


def test_an_empty_project_finishes_without_embedding(client, tmp_path, stub_embedding):
    """No chunks means no API call — indexing must short-circuit, not crash."""
    with client.websocket_connect("/api/projects/ws/embed") as ws:
        ws.send_json({"path": str(tmp_path)})
        events = _drain_steps(ws)

    assert events[-1]["step"] == "done"
    assert "path" not in stub_embedding, "nothing should have been stored"


def test_an_indexing_failure_is_reported_and_logged(client, tmp_project, monkeypatch, caplog):
    async def boom(texts, **kwargs):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(project_index, "embed_batches", boom)
    with caplog.at_level("ERROR", logger="core.project_index"):
        with client.websocket_connect("/api/projects/ws/embed") as ws:
            ws.send_json({"path": str(tmp_project)})
            events = _drain_steps(ws)

    assert events[-1]["step"] == "error"
    assert "provider exploded" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records), "the traceback must be kept"


# ── Coder planning phase ─────────────────────────────────────────────────────


class _FakeStream:
    """Async-iterates OpenAI-shaped streaming deltas."""

    def __init__(self, tokens):
        self._tokens = list(tokens)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        delta = SimpleNamespace(content=self._tokens.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def test_planning_streams_tokens_then_the_full_plan(tmp_project, monkeypatch):
    """The right-hand panel fills live, so tokens must arrive individually."""
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini")

    async def fake_stream(client, **kwargs):
        return _FakeStream(["1. ", "read ", "main.py"])

    monkeypatch.setattr(coder_mod, "chat_stream", fake_stream)
    events = [e async for e in agent._plan("task", "TREE")]

    assert [e["chunk"] for e in events if e["type"] == "plan_chunk"] == ["1. ", "read ", "main.py"]
    assert events[-1] == {"type": "plan", "content": "1. read main.py"}


async def test_reasoning_models_plan_in_a_single_shot(tmp_project, monkeypatch):
    """o1-family streaming is unreliable, so that path must not be used."""
    agent = CoderAgent(str(tmp_project), "gpt-5.5-high-effort")

    async def fake_completion(client, **kwargs):
        assert "stream" not in kwargs
        return make_response(content="1. do the thing")

    monkeypatch.setattr(coder_mod, "chat_completion", fake_completion)
    events = [e async for e in agent._plan("task", "TREE")]

    assert events[-1] == {"type": "plan", "content": "1. do the thing"}


async def test_empty_stream_tokens_are_ignored(tmp_project, monkeypatch):
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini")

    async def fake_stream(client, **kwargs):
        return _FakeStream(["a", "", None, "b"])

    monkeypatch.setattr(coder_mod, "chat_stream", fake_stream)
    events = [e async for e in agent._plan("task", "TREE")]

    assert len([e for e in events if e["type"] == "plan_chunk"]) == 2
    assert events[-1]["content"] == "ab"


async def test_a_full_implement_run_emits_the_expected_event_sequence(tmp_project, monkeypatch):
    """plan -> tools -> summary -> validation -> done."""
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini")
    monkeypatch.setattr(coder_mod, "get_client", lambda timeout: object())

    async def fake_plan(prompt, tree):
        yield {"type": "plan", "content": "the plan"}

    async def fake_loop(messages, tools=None):
        yield {"type": "impl_done", "content": "the summary"}

    async def fake_validate(messages):
        yield {"type": "validation_result", "phase": "lint", "passed": True,
               "status": "passed", "output": ""}

    monkeypatch.setattr(agent, "_plan", fake_plan)
    monkeypatch.setattr(agent, "_tool_loop", fake_loop)
    monkeypatch.setattr(agent, "_validate", fake_validate)

    events = [e async for e in agent.execute("build it")]
    assert [e["type"] for e in events] == ["plan", "impl_done", "validation_result", "done"]
    assert events[-1]["summary"] == "the summary"


async def test_an_unexpected_failure_mid_run_is_logged_and_reported(
    tmp_project, monkeypatch, caplog
):
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini")
    monkeypatch.setattr(coder_mod, "get_client", lambda timeout: object())

    async def boom(prompt, tree):
        raise RuntimeError("something broke")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(agent, "_plan", boom)
    with caplog.at_level("ERROR", logger="agents.coder"):
        events = [e async for e in agent.execute("build it")]

    assert events[-1]["type"] == "error"
    assert "something broke" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records)
