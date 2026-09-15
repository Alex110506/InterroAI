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
