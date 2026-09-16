"""
The chat WebSocket adapter.

The pipeline it drives is tested transport-free in `test_session.py`; what is
left here is the socket contract — which frames are accepted, what a bad frame
produces, and whether a rejected frame closes the connection.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import agents.session as session
import api.chat as chat
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def _start(**overrides):
    payload = {
        "type": "start",
        "project_path": "/tmp/p",
        "project_index": {},
        "message": "hello",
    }
    payload.update(overrides)
    return payload


def test_an_unknown_model_is_rejected(client):
    """
    Auto-routing was removed, so the caller now names the model. An unknown id
    must fail here with a clear message rather than deep inside the OpenAI call.
    """
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(model="gpt-9-ultra"))
        event = ws.receive_json()
    assert event["type"] == "error"
    assert "Unknown model" in event["message"]
    assert "gpt-5.6-sol" in event["message"], "the error should list valid ids"


def test_the_removed_auto_sentinel_is_rejected(client):
    """A stale client still sending "auto" gets a real explanation."""
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(model="auto"))
        event = ws.receive_json()
    assert event["type"] == "error"
    assert "Unknown model" in event["message"]


def test_a_rejected_model_does_not_close_the_session(client):
    """The user should be able to pick a valid model and carry on."""
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(model="bogus"))
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "wat"})
        assert "Unknown message type" in ws.receive_json()["message"]


def test_an_unknown_message_type_is_reported(client):
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json({"type": "nonsense"})
        event = ws.receive_json()
    assert event["type"] == "error"
    assert "Unknown message type" in event["message"]


def test_the_default_model_is_a_known_id():
    assert chat._DEFAULT_MODEL in session.AVAILABLE_MODELS


def test_being_signed_out_reaches_the_user_as_a_clear_error(client):
    """
    End-to-end, through the real default gateway: the typed error raised in
    `core.remote.session` survives the session and the WebSocket handler
    intact — code and all, since the frontend routes on it back to sign-in.
    """
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(_start(model="gpt-5.6-sol"))
        event = ws.receive_json()
    assert event["type"] == "error"
    assert "Sign in" in event["message"]
    assert event["code"] == "not_signed_in"
