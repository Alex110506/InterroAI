"""The Web API's application factory."""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient

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
