"""
`/projects` over HTTP on real Postgres, row-level security included.

Needs the local stack: `pytest -m integration`. Driven through httpx's ASGI
transport rather than TestClient, so the app runs on the test's own event loop,
which is the loop its database connections belong to.
"""
from __future__ import annotations

import itertools
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from cloud.api.main import create_app
from cloud.api.services import Services
from cloud.api.tokens import TokenSigner
from cloud.db.accounts import PostgresAccounts
from cloud.db.projects import ProjectRepository
from cloud.db.session import anonymous_scope

pytestmark = pytest.mark.integration

_github_ids = itertools.count(50_000)


@pytest.fixture
async def api(app_sessions):
    signer = TokenSigner("p" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))
    accounts = PostgresAccounts(lambda: anonymous_scope(app_sessions))
    services = Services(
        signin=None,
        signer=signer,
        accounts=accounts,
        projects=ProjectRepository(app_sessions),
    )
    transport = httpx.ASGITransport(app=create_app(services))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield SimpleNamespace(client=client, signer=signer, accounts=accounts)


async def _headers(api, login: str) -> dict[str, str]:
    user = await api.accounts.upsert_github_user(next(_github_ids), login, None)
    return {"Authorization": f"Bearer {api.signer.access_token(user.id, user.login)}"}


async def test_a_user_creates_lists_reads_and_deletes_a_project(api):
    alice = await _headers(api, "alice")

    created = await api.client.post("/projects", json={"name": "  interroai  "}, headers=alice)
    assert created.status_code == 201
    project = created.json()
    assert project["name"] == "interroai"

    listed = await api.client.get("/projects", headers=alice)
    assert [entry["id"] for entry in listed.json()] == [project["id"]]
    assert (await api.client.get(f"/projects/{project['id']}", headers=alice)).json() == project

    deleted = await api.client.delete(f"/projects/{project['id']}", headers=alice)
    assert deleted.status_code == 204
    gone = await api.client.get(f"/projects/{project['id']}", headers=alice)
    assert gone.status_code == 404
    assert gone.json()["detail"]["code"] == "project_not_found"


async def test_another_users_project_does_not_exist_for_them(api):
    alice, bob = await _headers(api, "alice"), await _headers(api, "bob")
    project = (await api.client.post("/projects", json={"name": "private"}, headers=alice)).json()

    assert (await api.client.get(f"/projects/{project['id']}", headers=bob)).status_code == 404
    assert (await api.client.delete(f"/projects/{project['id']}", headers=bob)).status_code == 404
    assert (await api.client.get("/projects", headers=bob)).json() == []
    assert (await api.client.get(f"/projects/{project['id']}", headers=alice)).status_code == 200


async def test_projects_need_a_signed_in_user(api):
    assert (await api.client.get("/projects")).status_code == 401
    assert (await api.client.post("/projects", json={"name": "x"})).status_code == 401


async def test_a_project_needs_a_real_name(api):
    alice = await _headers(api, "alice")
    response = await api.client.post("/projects", json={"name": "   "}, headers=alice)
    assert response.status_code == 422


async def test_a_malformed_project_id_is_rejected(api):
    alice = await _headers(api, "alice")
    assert (await api.client.get("/projects/not-a-uuid", headers=alice)).status_code == 422
