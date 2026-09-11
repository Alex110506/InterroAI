"""
Credential storage. `keyring` is replaced with an in-memory double so the test
run never reads from or writes to the developer's real OS keychain.
"""
from __future__ import annotations

import pytest
from keyring.errors import PasswordDeleteError

import core.security as security


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
    assert security.retrieve_openai_key() is None


def test_store_then_retrieve_round_trips(fake_keyring):
    security.store_openai_key("sk-abc123")
    assert security.retrieve_openai_key() == "sk-abc123"


def test_storing_twice_overwrites(fake_keyring):
    security.store_openai_key("sk-first")
    security.store_openai_key("sk-second")
    assert security.retrieve_openai_key() == "sk-second"


def test_key_is_namespaced_by_service_and_account(fake_keyring):
    """Guards against a rename silently orphaning an existing user's key."""
    security.store_openai_key("sk-abc123")
    assert fake_keyring.store == {("interroai", "openai_api_key"): "sk-abc123"}


def test_delete_removes_the_key(fake_keyring):
    security.store_openai_key("sk-abc123")
    security.delete_openai_key()
    assert security.retrieve_openai_key() is None


def test_deleting_a_missing_key_is_a_no_op(fake_keyring):
    """
    Deliberate, narrow swallow: `PasswordDeleteError` only ever means "there
    was nothing to delete", which is the caller's desired end state anyway.
    """
    security.delete_openai_key()  # must not raise
    assert security.retrieve_openai_key() is None


def test_delete_does_not_swallow_unexpected_errors(monkeypatch, fake_keyring):
    """Only the 'already absent' case is tolerated — real faults propagate."""

    def explode(service, account):
        raise RuntimeError("keychain locked")

    monkeypatch.setattr(fake_keyring, "delete_password", explode)
    with pytest.raises(RuntimeError, match="keychain locked"):
        security.delete_openai_key()


def test_empty_string_is_falsy_for_callers(fake_keyring):
    """
    `get_client` branches on truthiness, not on `is None`, so an empty stored
    value must still be treated as "no key".
    """
    security.store_openai_key("")
    assert not security.retrieve_openai_key()
