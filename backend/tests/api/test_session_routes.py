"""The runtime's `/api/session` routes, in local mode and in cloud mode."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from core import providers
from core.remote.session import CloudSession
from main import app

API = "https://api.example"
VERIFIER = "v" * 43


def _fake_cloud_api(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/auth/token":
        if json.loads(request.content).get("code") == "good-code":
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "token_type": "Bearer",
                    "expires_in": 900,
                },
            )
        return httpx.Response(
            400, json={"detail": {"code": "invalid_grant", "message": "The code has expired."}}
        )
    if request.url.path == "/auth/logout":
        return httpx.Response(204)
    if request.url.path == "/me":
        return httpx.Response(
            200, json={"id": "user-1", "login": "octocat", "avatar_url": "https://avatars/1"}
        )
    return httpx.Response(404)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def cloud(monkeypatch, cloud_mode, fake_keyring):
    session = CloudSession(API, transport=httpx.MockTransport(_fake_cloud_api))
    monkeypatch.setattr(providers, "cloud_session", lambda: session)
    return session


# ── Local mode ───────────────────────────────────────────────────────────────


def test_a_local_runtime_says_so(client):
    assert client.get("/api/session").json() == {
        "mode": "local",
        "api_url": None,
        "signed_in": False,
        "login": None,
        "avatar_url": None,
    }


def test_signing_in_is_refused_in_local_mode(client):
    response = client.put("/api/session", json={"code": "good-code", "code_verifier": VERIFIER})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "local_mode"


def test_the_settings_report_local_mode(client, fake_keyring):
    assert client.get("/api/settings").json()["mode"] == "local"


# ── Cloud mode ───────────────────────────────────────────────────────────────


def test_a_cloud_runtime_starts_signed_out(client, cloud):
    assert client.get("/api/session").json() == {
        "mode": "cloud",
        "api_url": API,
        "signed_in": False,
        "login": None,
        "avatar_url": None,
    }


def test_the_apps_login_code_signs_the_runtime_in(client, cloud, fake_keyring):
    response = client.put("/api/session", json={"code": "good-code", "code_verifier": VERIFIER})

    assert response.status_code == 200
    assert response.json()["login"] == "octocat"
    assert "refresh" not in response.text, "tokens stay in this process"
    assert fake_keyring[("interroai", f"cloud_refresh_token:{API}")] == "refresh"
    assert client.get("/api/session").json()["signed_in"] is True


def test_a_bad_login_code_is_a_400_carrying_the_apis_code(client, cloud):
    response = client.put("/api/session", json={"code": "stale-code", "code_verifier": VERIFIER})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_grant"


def test_signing_out_forgets_the_session(client, cloud):
    client.put("/api/session", json={"code": "good-code", "code_verifier": VERIFIER})

    assert client.delete("/api/session").status_code == 204
    assert client.get("/api/session").json()["signed_in"] is False


def test_the_settings_report_cloud_mode(client, cloud):
    assert client.get("/api/settings").json()["mode"] == "cloud"
