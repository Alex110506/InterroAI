"""What every Cloud API request goes through: request ids, access logs and the body limit."""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fakes.accounts import InMemoryAccounts
from fastapi.testclient import TestClient

from cloud.api.main import create_app
from cloud.api.services import Limits, Services
from cloud.api.tokens import TokenSigner


@pytest.fixture
def world():
    accounts = InMemoryAccounts()
    signer = TokenSigner("m" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))
    services = Services(
        signin=None,
        signer=signer,
        accounts=accounts,
        projects=None,
        limits=Limits(max_request_bytes=1_000),
    )
    client = TestClient(create_app(services))
    return SimpleNamespace(client=client, accounts=accounts, signer=signer)


# ── Request ids ──────────────────────────────────────────────────────────────


def test_every_response_carries_its_own_request_id(world):
    first, second = world.client.get("/health"), world.client.get("/health")

    assert len(first.headers["x-request-id"]) == 32
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


def test_a_request_id_from_the_proxy_is_kept(world):
    response = world.client.get("/health", headers={"X-Request-ID": "front-door-1234"})
    assert response.headers["x-request-id"] == "front-door-1234"


def test_a_request_id_that_is_not_an_id_is_replaced(world):
    response = world.client.get("/health", headers={"X-Request-ID": "not an id at all"})
    assert response.headers["x-request-id"] != "not an id at all"


# ── Access log ───────────────────────────────────────────────────────────────


def test_each_request_is_logged_once_with_its_outcome_and_user(world, caplog):
    user = asyncio.run(world.accounts.upsert_github_user(1, "octocat", None))
    token = world.signer.access_token(user.id, user.login)

    with caplog.at_level(logging.INFO, logger="cloud.api.access"):
        response = world.client.get("/me", headers={"Authorization": f"Bearer {token}"})

    [record] = [r for r in caplog.records if r.name == "cloud.api.access"]
    assert (record.method, record.path, record.status) == ("GET", "/me", 200)
    assert record.request_id == response.headers["x-request-id"]
    assert record.user_id == user.id


def test_health_probes_do_not_fill_the_log(world, caplog):
    with caplog.at_level(logging.INFO, logger="cloud.api.access"):
        world.client.get("/health")
    assert [r for r in caplog.records if r.name == "cloud.api.access"] == []


# ── Body limit ───────────────────────────────────────────────────────────────


def test_an_oversized_body_is_refused_before_it_is_read(world):
    response = world.client.post(
        "/auth/token",
        content=b'{"grant_type": "refresh_token", "refresh_token": "' + b"x" * 2_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"


def test_a_body_that_does_not_declare_its_length_is_refused(world):
    def chunks():
        yield b'{"grant_type": "refresh_token", "refresh_token": "x"}'

    response = world.client.post(
        "/auth/token", content=chunks(), headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 411
    assert response.json()["detail"]["code"] == "length_required"
