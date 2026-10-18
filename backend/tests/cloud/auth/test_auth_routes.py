"""
The auth routes over HTTP, on in-memory accounts and a fake Auth0 tenant.

Signing in itself happens between the app and Auth0, so the API has two routes
left: `/auth/config`, which tells the app where to sign in, and `/me`. What is
tested here is what HTTP adds to `IdentityService` (`test_identity.py`): status
codes and error bodies, for every kind of token `current_user` can meet.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fakes.accounts import InMemoryAccounts
from fakes.auth0 import AUDIENCE, CLIENT_ID, DOMAIN, FakeAuth0
from fastapi.testclient import TestClient

from cloud.api.main import create_app
from cloud.api.services import Services, build_services
from cloud.auth.auth0 import Auth0Config
from cloud.auth.identity import IdentityService
from cloud.settings import ApiSettings


@pytest.fixture
def api():
    tenant = (
        FakeAuth0()
        .add_user(583231, "Alex110506", "https://avatars.example/583231")
        .add_user(777, "stranger")
    )
    accounts = InMemoryAccounts()
    services = Services(identity=tenant.identity(accounts), projects=None, auth0=tenant.config)
    client = TestClient(create_app(services))
    return SimpleNamespace(client=client, tenant=tenant, accounts=accounts)


def test_the_app_learns_where_to_sign_in_without_being_signed_in(api):
    response = api.client.get("/auth/config")

    assert response.status_code == 200
    assert response.json() == {"domain": DOMAIN, "client_id": CLIENT_ID, "audience": AUDIENCE}


def test_me_names_the_signed_in_github_account(api):
    response = api.client.get("/me", headers=api.tenant.bearer(583231))

    assert response.status_code == 200
    body = response.json()
    assert (body["login"], body["avatar_url"]) == ("Alex110506", "https://avatars.example/583231")
    assert body["id"] == api.accounts.users[body["id"]].id


def test_me_follows_a_github_rename(api):
    api.client.get("/me", headers=api.tenant.bearer(583231))
    api.tenant.add_user(583231, "Alex110506", "https://avatars.example/new")

    assert api.client.get("/me", headers=api.tenant.bearer(583231)).json()["avatar_url"] == (
        "https://avatars.example/new"
    )


def test_no_token_is_not_signed_in(api):
    response = api.client.get("/me")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "not_signed_in"
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_bad_token_is_an_invalid_token_so_the_app_refreshes(api):
    response = api.client.get(
        "/me", headers={"Authorization": f"Bearer {api.tenant.token(583231, expires_in=-60)}"}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_token"


def test_an_account_off_the_allowlist_is_refused_with_a_reason(api):
    response = api.client.get("/me", headers=api.tenant.bearer(777))

    assert response.status_code == 403
    assert response.json()["detail"] == {
        "code": "access_denied",
        "message": "This GitHub account is not allowed to use InterroAI.",
    }


def test_auth0_down_on_a_first_request_is_a_503_not_a_sign_out(api):
    api.tenant.down = True
    response = api.client.get("/me", headers=api.tenant.bearer(583231))

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "auth_unavailable"


async def test_production_wiring_takes_auth0_from_the_settings():
    settings = ApiSettings(
        _env_file=None,
        database_url="postgresql+asyncpg://app:pw@127.0.0.1:1/none",
        servicebus_connection_string=(
            "Endpoint=sb://127.0.0.1;SharedAccessKeyName=k;SharedAccessKey=v;"
            "UseDevelopmentEmulator=true;"
        ),
        blob_connection_string=(
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=a2V5;"
            "BlobEndpoint=http://127.0.0.1:1/devstoreaccount1;"
        ),
        openai_api_key="unused",
        auth0_domain="tenant.auth0.example",
        auth0_client_id="the-desktop-app",
        auth0_audience="https://api.example",
        allowed_github_logins="Alex110506",
    )
    services = build_services(settings)
    try:
        assert services.auth0 == Auth0Config(
            domain="tenant.auth0.example",
            client_id="the-desktop-app",
            audience="https://api.example",
        )
        assert isinstance(services.identity, IdentityService)
    finally:
        await services.close()
