"""The Web API's application factory."""
from __future__ import annotations

import asyncio
import logging

from fastapi.testclient import TestClient

from cloud.api import services as services_module
from cloud.api.main import VERSION, create_app
from cloud.api.services import Services


def _services(*startup) -> Services:
    return Services(signin=None, signer=None, accounts=None, projects=None, startup=list(startup))


def test_health_reports_ok():
    body = TestClient(create_app()).get("/health").json()
    assert body == {"status": "ok", "version": VERSION}


def test_building_the_app_needs_no_configuration(monkeypatch):
    """Importing and constructing must never require a `.env`."""
    monkeypatch.delenv("INTERROAI_DATABASE_URL", raising=False)
    assert create_app().title == "InterroAI Cloud API"


def test_startup_steps_run_as_the_app_starts():
    ran = []

    async def create_upload_container():
        ran.append("upload container")

    with TestClient(create_app(_services(create_upload_container))):
        assert ran == ["upload container"]


def test_a_failing_startup_step_does_not_stop_the_app(caplog):
    """With storage down, sign-in and everything else should still work."""

    async def storage_is_down():
        raise ConnectionError("Azurite is not running")

    with caplog.at_level(logging.WARNING, logger="cloud.api.services"):
        with TestClient(create_app(_services(storage_is_down))) as client:
            assert client.get("/health").status_code == 200

    assert "startup step failed" in caplog.text


def test_a_startup_step_that_never_answers_does_not_hold_the_app_back(caplog, monkeypatch):
    """
    An unreachable service does not fail fast: the Azure SDK retries it for
    minutes. Waiting for that means the app misses its first health check.
    """
    monkeypatch.setattr(services_module, "_STARTUP_STEP_TIMEOUT_SECONDS", 0.01)

    async def storage_never_answers():
        await asyncio.sleep(30)

    with caplog.at_level(logging.WARNING, logger="cloud.api.services"):
        with TestClient(create_app(_services(storage_never_answers))) as client:
            assert client.get("/health").status_code == 200

    assert "startup step timed out" in caplog.text
