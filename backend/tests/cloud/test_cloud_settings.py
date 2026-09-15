"""Cloud settings: what each process requires, and how the risky values are handled."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cloud.settings import ApiSettings, WorkerSettings

_WORKER = {
    "database_url": "postgresql+asyncpg://app:pw@localhost:5432/interroai",
    "servicebus_connection_string": "Endpoint=sb://localhost;UseDevelopmentEmulator=true;",
    "blob_connection_string": "UseDevelopmentStorage=true",
    "openai_api_key": "sk-test-platform-key",
}
_API = {
    **_WORKER,
    "github_client_id": "client-id",
    "github_client_secret": "client-secret",
    "jwt_secret": "x" * 32,
}


@pytest.fixture(autouse=True)
def no_ambient_configuration(monkeypatch):
    """A developer's shell or `.env` must not decide what these tests see."""
    for name in list(_API) + ["allowed_github_logins"]:
        monkeypatch.delenv(f"INTERROAI_{name.upper()}", raising=False)


def _api(**overrides) -> ApiSettings:
    return ApiSettings(_env_file=None, **{**_API, **overrides})


def test_the_worker_needs_no_sign_in_secrets():
    settings = WorkerSettings(_env_file=None, **_WORKER)
    assert settings.servicebus_queue == "index-jobs"


def test_a_missing_database_url_fails_at_start_up():
    values = {key: value for key, value in _WORKER.items() if key != "database_url"}
    with pytest.raises(ValidationError):
        WorkerSettings(_env_file=None, **values)


def test_the_allowlist_is_case_insensitive_and_trimmed():
    settings = _api(allowed_github_logins="Alex110506, Someone-Else ,")
    assert settings.allowlist == frozenset({"alex110506", "someone-else"})


def test_an_empty_allowlist_admits_nobody():
    """Fail closed: a forgotten setting must not open sign-up on the platform key."""
    assert _api().allowlist == frozenset()


def test_a_short_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        _api(jwt_secret="too-short")


def test_secrets_never_appear_in_a_printed_settings_object():
    settings = _api()
    printed = repr(settings) + str(settings)
    assert "sk-test-platform-key" not in printed
    assert "client-secret" not in printed
