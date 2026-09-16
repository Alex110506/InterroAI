"""Runtime settings: where the Cloud API is, and how the app locks the runtime down."""
from __future__ import annotations

import pytest

from core.settings import RuntimeSettings


def test_the_api_url_defaults_to_the_local_stack(monkeypatch):
    monkeypatch.delenv("INTERROAI_API_URL", raising=False)
    assert RuntimeSettings(_env_file=None).api_url == "http://localhost:8080"


def test_the_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("INTERROAI_API_URL", "https://api.interroai.example")
    assert RuntimeSettings(_env_file=None).api_url == "https://api.interroai.example"


def test_the_launch_token_comes_from_the_app_and_is_never_printed(monkeypatch):
    monkeypatch.setenv("INTERROAI_LAUNCH_TOKEN", "t0k3n-from-the-app")

    settings = RuntimeSettings(_env_file=None)

    assert settings.launch_token.get_secret_value() == "t0k3n-from-the-app"
    assert "t0k3n-from-the-app" not in repr(settings)


@pytest.mark.parametrize("value", [None, ""])
def test_without_a_launch_token_there_is_none(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("INTERROAI_LAUNCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("INTERROAI_LAUNCH_TOKEN", value)
    assert RuntimeSettings(_env_file=None).launch_token is None


def test_allowed_origins_are_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("INTERROAI_ALLOWED_ORIGINS", "http://localhost:5173, null")
    assert RuntimeSettings(_env_file=None).origin_list == ["http://localhost:5173", "null"]
