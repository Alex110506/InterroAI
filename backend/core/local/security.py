"""Secure credential storage using the OS native keychain via keyring."""

import keyring
from keyring.errors import PasswordDeleteError

_SERVICE = "interroai"
_ACCOUNT_OPENAI = "openai_api_key"


def store_openai_key(key: str) -> None:
    """Persist the OpenAI API key in the OS keychain (macOS Keychain on Mac)."""
    keyring.set_password(_SERVICE, _ACCOUNT_OPENAI, key)


def retrieve_openai_key() -> str | None:
    """Return the stored OpenAI API key, or None if not set."""
    return keyring.get_password(_SERVICE, _ACCOUNT_OPENAI)


def delete_openai_key() -> None:
    """Remove the OpenAI API key from the keychain."""
    try:
        keyring.delete_password(_SERVICE, _ACCOUNT_OPENAI)
    except PasswordDeleteError:
        pass
