"""
The remaining HTTP surface: health, settings, and the supervisor that bridges
Coder Agent events onto the WebSocket.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import agents.supervisor as supervisor
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ───────────────────────────────────────────────────────────────────


def test_health_reports_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == app.version


def test_health_needs_no_credentials(client):
    """It is the readiness probe a remote client polls before connecting."""
    assert client.get("/health").status_code == 200


def test_all_routers_are_mounted(client, fake_keyring):
    """
    Asserted behaviourally rather than by inspecting `app.routes`: FastAPI wraps
    included routers in a private type, and a "not 404" is the property we
    actually care about.
    """
    assert client.get("/health").status_code != 404
    assert client.get("/api/settings").status_code != 404
    assert client.post("/api/projects/index", json={}).status_code != 404


def test_the_chat_websocket_is_mounted(client):
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json({"type": "ping-unknown"})
        assert ws.receive_json()["type"] == "error"


def test_the_console_entry_point_exists():
    """`pyproject.toml` declares `interroai-backend = "main:run"`."""
    import main

    assert callable(main.run)


# ── Settings ─────────────────────────────────────────────────────────────────


def test_the_user_name_round_trips(client):
    client.post("/api/settings", json={"name": "Alex"})
    assert client.get("/api/settings").json()["name"] == "Alex"


def test_the_settings_carry_nothing_but_the_name(client):
    """Model access is the platform's: this machine stores no credential to report."""
    assert set(client.get("/api/settings").json()) == {"name"}


def test_an_over_long_name_is_rejected(client):
    assert client.post("/api/settings", json={"name": "x" * 200}).status_code == 422


def test_settings_accept_an_empty_body(client, fake_keyring):
    assert client.post("/api/settings", json={}).status_code == 200


# ── Supervisor ───────────────────────────────────────────────────────────────


class RecordingSocket:
    def __init__(self, fail_after: int | None = None):
        self.sent: list[dict] = []
        self._fail_after = fail_after

    async def send_json(self, payload):
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            raise RuntimeError("client went away")
        self.sent.append(payload)


def _stub_agent(monkeypatch, events):
    class StubAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def execute(self, prompt):
            for event in events:
                yield event

    monkeypatch.setattr("agents.coder.CoderAgent", StubAgent)


async def test_every_agent_event_is_forwarded(monkeypatch):
    events = [
        {"type": "plan", "content": "the plan"},
        {"type": "tool_call", "tool": "read_file", "args": {}},
        {"type": "done", "summary": "finished"},
    ]
    _stub_agent(monkeypatch, events)
    socket = RecordingSocket()
    await supervisor.run("task", "/proj", "gpt-5.6-sol", socket)
    assert socket.sent == events


async def test_a_disconnected_client_stops_the_stream(monkeypatch):
    """
    The user closed the window; continuing to push events would raise on every
    remaining one for no benefit.
    """
    _stub_agent(monkeypatch, [{"type": "plan"}, {"type": "tool_call"}, {"type": "done"}])
    socket = RecordingSocket(fail_after=1)
    await supervisor.run("task", "/proj", "gpt-5.6-sol", socket)
    assert len(socket.sent) == 1


async def test_the_supervisor_works_without_a_socket(monkeypatch):
    """Keeps the pipeline usable from a test harness or any non-socket caller."""
    _stub_agent(monkeypatch, [{"type": "done", "summary": "ok"}])
    await supervisor.run("task", "/proj", "gpt-5.6-sol", None)


async def test_the_intent_is_passed_through(monkeypatch):
    captured = {}

    class StubAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def execute(self, prompt):
            yield {"type": "done", "summary": ""}

    monkeypatch.setattr("agents.coder.CoderAgent", StubAgent)
    await supervisor.run("task", "/proj", "gpt-5.6-sol", None, intent="answer")
    assert captured["intent"] == "answer"
    assert captured["model"] == "gpt-5.6-sol"
