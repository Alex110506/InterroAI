"""
End-to-end paths through the two WebSocket pipelines.

The unit tests elsewhere cover each component in isolation; these drive the
actual sockets so the routing between them — which intent leads where, and what
the client receives in what order — is pinned down too.
"""
from __future__ import annotations

import pytest
from conftest import FakeGateway, make_response
from fastapi.testclient import TestClient

import agents.session as session
import core.index.adapters.chroma as chroma
import core.index.indexer as indexer
import core.index.semantic_index as semantic_index
import core.workspace.project_index as project_index
from agents.coder import CoderAgent
from core.index.embeddings import EmbeddedBatch
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def captured_supervisor(monkeypatch):
    """Replace the supervisor so no agent or network work actually happens."""
    calls: list[dict] = []

    async def fake_stream(
        prompt, project_path, model, intent="implement", history=None, gateway=None
    ):
        calls.append(
            {"prompt": prompt, "project_path": project_path, "model": model, "intent": intent}
        )
        yield {"type": "done", "summary": "stubbed"}

    monkeypatch.setattr(session.supervisor, "stream", fake_stream)
    return calls


def _fix_intent(monkeypatch, action):
    async def fake_classify(message, index, history=None, *, gateway=None):
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
        stored.setdefault("chunks", []).extend(chunks)

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    monkeypatch.setattr(chroma, "store_chunks", fake_store)
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

    monkeypatch.setattr(indexer, "embed_batches", boom)
    with caplog.at_level("ERROR", logger="core.index.indexer"):
        with client.websocket_connect("/api/projects/ws/embed") as ws:
            ws.send_json({"path": str(tmp_project)})
            events = _drain_steps(ws)

    assert events[-1]["step"] == "error"
    assert "provider exploded" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records), "the traceback must be kept"


async def test_a_search_after_indexing_reads_the_code_back_from_disk(
    tmp_project, isolated_chroma, monkeypatch
):
    """
    The whole of design B in one pass: index the project through the real
    local index, then search it through the real agent. The index stored no
    text, so every line of code in the result was read from the working tree.
    """
    vector = [0.1, 0.2, 0.3, 0.4]

    async def fake_batches(texts, **kwargs):
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[vector for _ in texts])

    async def fake_query(texts, **kwargs):
        return [vector for _ in texts]

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    monkeypatch.setattr(semantic_index, "embed_texts", fake_query)

    events = [e async for e in project_index.embed_project(tmp_project)]
    assert events[-1]["step"] == "done"

    out = await CoderAgent(str(tmp_project), "gpt-5.4-mini")._search_semantic("greeting", n=10)
    assert "class Greeter" in out
    assert "stale" not in out


# ── Coder planning phase─────────────────────────────────────────────────────


async def test_planning_streams_tokens_then_the_full_plan(tmp_project):
    """The right-hand panel fills live, so tokens must arrive individually."""
    gateway = FakeGateway(streams=[["1. ", "read ", "main.py"]])
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", gateway=gateway)
    events = [e async for e in agent._plan("task", "TREE")]

    assert [e["chunk"] for e in events if e["type"] == "plan_chunk"] == ["1. ", "read ", "main.py"]
    assert events[-1] == {"type": "plan", "content": "1. read main.py"}


async def test_reasoning_models_plan_in_a_single_shot(tmp_project):
    """
    o1-family streaming is unreliable, so that path must not be used — the
    gateway has no stream queued, and would fail the test if asked for one.
    """
    gateway = FakeGateway([make_response(content="1. do the thing")])
    agent = CoderAgent(str(tmp_project), "gpt-5.5-high-effort", gateway=gateway)
    events = [e async for e in agent._plan("task", "TREE")]

    assert events[-1] == {"type": "plan", "content": "1. do the thing"}


async def test_empty_stream_tokens_are_ignored(tmp_project):
    gateway = FakeGateway(streams=[["a", "", None, "b"]])
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini", gateway=gateway)
    events = [e async for e in agent._plan("task", "TREE")]

    assert len([e for e in events if e["type"] == "plan_chunk"]) == 2
    assert events[-1]["content"] == "ab"


async def test_a_full_implement_run_emits_the_expected_event_sequence(tmp_project, monkeypatch):
    """plan -> tools -> summary -> validation -> done."""
    agent = CoderAgent(str(tmp_project), "gpt-5.4-mini")

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

    async def boom(prompt, tree):
        raise RuntimeError("something broke")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(agent, "_plan", boom)
    with caplog.at_level("ERROR", logger="agents.coder"):
        events = [e async for e in agent.execute("build it")]

    assert events[-1]["type"] == "error"
    assert "something broke" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records)
