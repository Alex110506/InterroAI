"""
The launch token, and the CORS policy around it.

Each app is built from explicit settings rather than the environment, so every
test says which runtime it is: one started by the Electron app (token set) or
one started by hand (no token).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from core.settings import RuntimeSettings
from main import create_app

TOKEN = "a" * 64
APP_ORIGIN = "http://localhost:5173"


def _client(**settings) -> TestClient:
    return TestClient(create_app(RuntimeSettings(_env_file=None, **settings)))


@pytest.fixture
def guarded():
    return _client(launch_token=TOKEN, allowed_origins=APP_ORIGIN)


def test_a_runtime_started_by_hand_needs_no_token():
    assert _client().get("/api/session").status_code == 200


def test_a_request_without_the_token_is_refused(guarded):
    response = guarded.get("/api/session")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "launch_token_required"


def test_a_request_with_the_wrong_token_is_refused(guarded):
    response = guarded.get("/api/session", headers={"X-Interroai-Token": "b" * 64})
    assert response.status_code == 401


def test_a_request_with_the_token_is_answered(guarded):
    response = guarded.get("/api/session", headers={"X-Interroai-Token": TOKEN})
    assert response.status_code == 200


def test_the_health_check_stays_open(guarded):
    assert guarded.get("/health").status_code == 200


def test_a_token_in_the_query_string_is_not_enough_for_plain_http(guarded):
    """URLs end up in logs and history; only WebSockets, which cannot send headers, use one."""
    assert guarded.get(f"/api/session?token={TOKEN}").status_code == 401


def test_a_websocket_without_the_token_is_refused_at_the_handshake(guarded):
    with pytest.raises(WebSocketDisconnect) as refused:
        with guarded.websocket_connect("/api/chat/ws"):
            pass
    assert refused.value.code == 4401


def test_a_websocket_with_the_token_connects(guarded):
    with guarded.websocket_connect(f"/api/chat/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "not-a-real-frame"})
        assert ws.receive_json()["type"] == "error"


def test_the_apps_origin_passes_the_cors_preflight(guarded):
    response = guarded.options(
        "/api/session",
        headers={
            "Origin": APP_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-interroai-token",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == APP_ORIGIN


def test_any_other_origin_does_not(guarded):
    response = guarded.options(
        "/api/session",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_a_refusal_still_carries_cors_headers_so_the_app_can_read_it(guarded):
    response = guarded.get("/api/session", headers={"Origin": APP_ORIGIN})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == APP_ORIGIN
