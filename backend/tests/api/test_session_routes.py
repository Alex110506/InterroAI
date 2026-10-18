"""The runtime's `/api/session` routes: where to sign in, and who is signed in."""
from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from core import providers
from core.remote.session import CloudSession
from main import app

API = "https://api.example"
AUTH0 = "tenant.auth0.example"
VERIFIER = "v" * 43
REDIRECT = "http://127.0.0.1:47213/callback"
AUTH0_CONFIG = {"domain": AUTH0, "client_id": "desktop-app", "audience": API}


def _fake_cloud(request: httpx.Request) -> httpx.Response:
    """The Cloud API and its Auth0 tenant, by host."""
    if request.url.host == AUTH0:
        form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
        if request.url.path == "/oauth/revoke":
            return httpx.Response(200)
        if form.get("code") == "good-code" and form.get("redirect_uri") == REDIRECT:
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
            403, json={"error": "invalid_grant", "error_description": "Invalid authorization code"}
        )
    if request.url.path == "/auth/config":
        return httpx.Response(200, json=AUTH0_CONFIG)
    if request.url.path == "/me":
        return httpx.Response(
            200, json={"id": "user-1", "login": "octocat", "avatar_url": "https://avatars/1"}
        )
    return httpx.Response(404)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def cloud(monkeypatch, cloud_api, fake_keyring):
    session = CloudSession(API, transport=httpx.MockTransport(_fake_cloud))
    monkeypatch.setattr(providers, "cloud_session", lambda: session)
    return session


def _sign_in(client, code: str = "good-code"):
    return client.put(
        "/api/session", json={"code": code, "code_verifier": VERIFIER, "redirect_uri": REDIRECT}
    )


def test_a_signed_out_runtime_says_where_to_sign_in(client, cloud):
    assert client.get("/api/session").json() == {
        "api_url": API,
        "signed_in": False,
        "login": None,
        "avatar_url": None,
        "auth0": AUTH0_CONFIG,
    }


def test_a_signed_out_runtime_with_the_cloud_unreachable_still_answers(client, monkeypatch):
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    session = CloudSession(API, transport=httpx.MockTransport(unreachable))
    monkeypatch.setattr(providers, "cloud_session", lambda: session)

    body = client.get("/api/session").json()
    assert (body["signed_in"], body["auth0"]) == (False, None)


def test_the_apps_authorization_code_signs_the_runtime_in(client, cloud, fake_keyring):
    response = _sign_in(client)

    assert response.status_code == 200
    assert response.json()["login"] == "octocat"
    assert "refresh" not in response.text, "tokens stay in this process"
    assert fake_keyring[("interroai", f"cloud_refresh_token:{API}")] == "refresh"
    assert client.get("/api/session").json()["signed_in"] is True


def test_a_bad_authorization_code_is_a_400_carrying_auth0s_code(client, cloud):
    response = _sign_in(client, code="stale-code")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_grant"


def test_signing_out_forgets_the_session(client, cloud):
    _sign_in(client)

    assert client.delete("/api/session").status_code == 204
    assert client.get("/api/session").json()["signed_in"] is False
