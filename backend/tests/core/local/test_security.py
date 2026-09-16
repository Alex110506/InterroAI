"""
Credential storage. `keyring` is replaced with an in-memory double so the test
run never reads from or writes to the developer's real OS keychain.

One credential lives here now: the cloud's refresh token, one per Cloud API URL.
The platform holds the model key, so this machine stores none.
"""
from __future__ import annotations

import pytest
from keyring.errors import PasswordDeleteError

import core.local.security as security

API = "https://api.interroai.example"
OTHER_API = "http://localhost:8080"


class FakeKeyring:
    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service, account, password):
        self.store[(service, account)] = password

    def get_password(self, service, account):
        return self.store.get((service, account))

    def delete_password(self, service, account):
        if (service, account) not in self.store:
            raise PasswordDeleteError("not found")
        del self.store[(service, account)]


@pytest.fixture
def fake_keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(security, "keyring", fake)
    return fake


def test_retrieve_returns_none_when_nothing_is_stored(fake_keyring):
    assert security.retrieve_refresh_token(API) is None


def test_store_then_retrieve_round_trips(fake_keyring):
    security.store_refresh_token(API, "refresh-abc123")
    assert security.retrieve_refresh_token(API) == "refresh-abc123"


def test_storing_twice_overwrites(fake_keyring):
    """Refresh tokens rotate on every use, so the newest must win."""
    security.store_refresh_token(API, "refresh-first")
    security.store_refresh_token(API, "refresh-second")
    assert security.retrieve_refresh_token(API) == "refresh-second"


def test_each_api_url_keeps_its_own_token(fake_keyring):
    """A token is only valid at the API that issued it."""
    security.store_refresh_token(API, "for-the-cloud")
    security.store_refresh_token(OTHER_API, "for-the-local-stack")

    assert security.retrieve_refresh_token(API) == "for-the-cloud"
    assert security.retrieve_refresh_token(OTHER_API) == "for-the-local-stack"


def test_a_trailing_slash_is_the_same_api(fake_keyring):
    """Otherwise the app would look signed out after a change that changed nothing."""
    security.store_refresh_token(API, "refresh-abc123")
    assert security.retrieve_refresh_token(f"{API}/") == "refresh-abc123"


def test_the_token_is_namespaced_by_service_and_account(fake_keyring):
    """Guards against a rename silently orphaning an existing user's session."""
    security.store_refresh_token(API, "refresh-abc123")
    assert fake_keyring.store == {("interroai", f"cloud_refresh_token:{API}"): "refresh-abc123"}


def test_delete_removes_the_token(fake_keyring):
    security.store_refresh_token(API, "refresh-abc123")
    security.delete_refresh_token(API)
    assert security.retrieve_refresh_token(API) is None


def test_deleting_a_missing_token_is_a_no_op(fake_keyring):
    """
    Deliberate, narrow swallow: `PasswordDeleteError` only ever means "there
    was nothing to delete", which is the caller's desired end state anyway.
    """
    security.delete_refresh_token(API)  # must not raise
    assert security.retrieve_refresh_token(API) is None


def test_delete_does_not_swallow_unexpected_errors(monkeypatch, fake_keyring):
    """Only the 'already absent' case is tolerated — real faults propagate."""

    def explode(service, account):
        raise RuntimeError("keychain locked")

    monkeypatch.setattr(fake_keyring, "delete_password", explode)
    with pytest.raises(RuntimeError, match="keychain locked"):
        security.delete_refresh_token(API)


def test_signing_out_of_one_api_leaves_the_other_alone(fake_keyring):
    security.store_refresh_token(API, "for-the-cloud")
    security.store_refresh_token(OTHER_API, "for-the-local-stack")

    security.delete_refresh_token(API)

    assert security.retrieve_refresh_token(API) is None
    assert security.retrieve_refresh_token(OTHER_API) == "for-the-local-stack"
