"""
The sign-in routes over HTTP, on in-memory accounts and a fake GitHub.

The flow's rules are tested on the service in `test_signin.py`. What is tested
here is what HTTP adds: redirects, the browser cookie, status codes, error
bodies and cache headers.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fakes.accounts import InMemoryAccounts
from fakes.github import FakeGitHub
from fastapi.testclient import TestClient

from cloud.api.main import create_app
from cloud.api.services import Services, build_services
from cloud.api.signin import SignInService
from cloud.api.tokens import TokenSigner
from cloud.settings import ApiSettings

VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
REDIRECT = "http://127.0.0.1:53682/callback"


@pytest.fixture
def api():
    accounts = InMemoryAccounts()
    signer = TokenSigner("r" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))
    signin = SignInService(
        accounts=accounts,
        github=FakeGitHub(),
        signer=signer,
        allowlist=frozenset({"alex110506"}),
        refresh_ttl=timedelta(days=30),
    )
    services = Services(signin=signin, signer=signer, accounts=accounts, projects=None)
    return SimpleNamespace(client=TestClient(create_app(services)), accounts=accounts)


def _location_query(response) -> dict[str, str]:
    query = parse_qs(urlsplit(response.headers["location"]).query)
    return {key: values[0] for key, values in query.items()}


def _start(client, **overrides):
    params = {
        "redirect_uri": REDIRECT,
        "code_challenge": CHALLENGE,
        "code_challenge_method": "S256",
        "state": "app-state",
    }
    return client.get("/auth/github/start", params=params | overrides, follow_redirects=False)


def _callback(client, started, **overrides):
    params = {"code": "gh-code", "state": _location_query(started)["state"]}
    return client.get(
        "/auth/github/callback", params=params | overrides, follow_redirects=False
    )


def _redeem(client, code: str, verifier: str = VERIFIER):
    return client.post(
        "/auth/token",
        json={"grant_type": "authorization_code", "code": code, "code_verifier": verifier},
    )


def _refresh(client, refresh_token: str):
    return client.post(
        "/auth/token", json={"grant_type": "refresh_token", "refresh_token": refresh_token}
    )


def _sign_in(client) -> dict:
    code = _location_query(_callback(client, _start(client)))["code"]
    response = _redeem(client, code)
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── The browser steps ────────────────────────────────────────────────────────


def test_start_redirects_to_github_and_sets_a_sign_in_cookie(api):
    response = _start(api.client)

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://github.example/login/oauth/authorize?")
    cookie = response.headers["set-cookie"].lower()
    assert cookie.startswith("interroai_signin=")
    for attribute in ("httponly", "path=/auth/github", "samesite=lax", "max-age=600"):
        assert attribute in cookie


def test_a_start_the_api_cannot_trust_gets_a_page_not_a_redirect(api):
    response = _start(api.client, redirect_uri="https://evil.example/callback")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("text/html")
    assert "location" not in response.headers


def test_the_callback_redirects_to_the_app_and_clears_the_cookie(api):
    response = _callback(api.client, _start(api.client))

    assert response.status_code == 302
    assert response.headers["location"].startswith(REDIRECT + "?")
    assert set(_location_query(response)) == {"code", "state"}
    assert "max-age=0" in response.headers["set-cookie"].lower()


def test_a_callback_in_a_browser_without_the_cookie_is_refused(api):
    started = _start(api.client)
    api.client.cookies.clear()

    response = _callback(api.client, started)

    assert response.status_code == 400
    assert "different browser" in response.text


# ── The app's token requests ─────────────────────────────────────────────────


def test_the_whole_sign_in_yields_a_working_access_token(api):
    tokens = _sign_in(api.client)

    assert tokens["token_type"] == "Bearer"
    assert tokens["expires_in"] == 900
    me = api.client.get("/me", headers=_bearer(tokens["access_token"]))
    assert me.status_code == 200
    assert me.json()["login"] == "Alex110506"


def test_token_responses_are_never_cached(api):
    granted = _refresh(api.client, _sign_in(api.client)["refresh_token"])
    refused = _redeem(api.client, "never-issued")

    assert granted.headers["cache-control"] == "no-store"
    assert refused.headers["cache-control"] == "no-store"


def test_a_bad_login_code_is_invalid_grant(api):
    response = _redeem(api.client, "never-issued")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_grant"


@pytest.mark.parametrize(
    "body",
    [{"grant_type": "authorization_code", "code": "c"}, {"grant_type": "refresh_token"}],
)
def test_a_token_request_missing_its_fields_is_invalid_request(api, body):
    response = api.client.post("/auth/token", json=body)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"


def test_an_unsupported_grant_type_is_rejected(api):
    assert api.client.post("/auth/token", json={"grant_type": "password"}).status_code == 422


def test_refresh_rotates_and_sign_out_ends_the_session(api):
    first = _sign_in(api.client)

    refreshed = _refresh(api.client, first["refresh_token"])
    assert refreshed.status_code == 200
    second = refreshed.json()

    signed_out = api.client.post("/auth/logout", json={"refresh_token": second["refresh_token"]})
    assert signed_out.status_code == 204
    assert _refresh(api.client, second["refresh_token"]).status_code == 400


def test_me_without_a_token_is_not_signed_in(api):
    response = api.client.get("/me")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"]["code"] == "not_signed_in"


def test_me_with_a_bad_token_is_invalid_token(api):
    response = api.client.get("/me", headers=_bearer("not-a-token"))

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_token"


# ── Wiring ───────────────────────────────────────────────────────────────────


async def test_services_build_from_settings_without_connecting_anywhere():
    settings = ApiSettings(
        _env_file=None,
        database_url="postgresql+asyncpg://app:pw@127.0.0.1:1/none",
        servicebus_connection_string="unused",
        blob_connection_string="unused",
        openai_api_key="unused",
        github_client_id="the-client-id",
        github_client_secret="the-client-secret",
        public_api_url="https://api.example/",
        jwt_secret="j" * 32,
        allowed_github_logins="Alex110506",
    )
    services = build_services(settings)
    try:
        started = services.signin.start(
            redirect_uri=REDIRECT,
            code_challenge=CHALLENGE,
            code_challenge_method="S256",
            app_state="app-state",
        )
        query = parse_qs(urlsplit(started.github_url).query)
        assert query["client_id"] == ["the-client-id"]
        assert query["redirect_uri"] == ["https://api.example/auth/github/callback"]
        assert services.secure_cookies
    finally:
        await services.close()
