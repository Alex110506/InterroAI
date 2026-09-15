"""The Web API's application factory."""
from __future__ import annotations

from fastapi.testclient import TestClient

from cloud.api.main import VERSION, create_app


def test_health_reports_ok():
    body = TestClient(create_app()).get("/health").json()
    assert body == {"status": "ok", "version": VERSION}


def test_building_the_app_needs_no_configuration(monkeypatch):
    """Importing and constructing must never require a `.env`."""
    monkeypatch.delenv("INTERROAI_DATABASE_URL", raising=False)
    assert create_app().title == "InterroAI Cloud API"
