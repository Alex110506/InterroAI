"""Runtime settings: which build the process is."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.settings import RuntimeSettings, get_runtime_settings


def test_the_default_build_is_fully_local(monkeypatch):
    monkeypatch.delenv("INTERROAI_MODE", raising=False)
    assert RuntimeSettings(_env_file=None).mode == "local"


def test_the_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("INTERROAI_MODE", "cloud")
    assert RuntimeSettings(_env_file=None).mode == "cloud"


def test_an_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("INTERROAI_MODE", "hybrid")
    with pytest.raises(ValidationError):
        RuntimeSettings(_env_file=None)


def test_the_test_suite_always_runs_the_local_build():
    """`conftest.py` pins it, so a developer's `.env` cannot reroute the suite."""
    assert get_runtime_settings().mode == "local"


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
