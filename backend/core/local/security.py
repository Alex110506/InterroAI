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


# ── Cloud session ────────────────────────────────────────────────────────────
# The InterroAI cloud's refresh token, one per Cloud API URL. The access token
# is never stored: it lasts fifteen minutes and lives in memory.

_ACCOUNT_REFRESH = "cloud_refresh_token"


def store_refresh_token(api_url: str, token: str) -> None:
    keyring.set_password(_SERVICE, _refresh_account(api_url), token)


def retrieve_refresh_token(api_url: str) -> str | None:
    return keyring.get_password(_SERVICE, _refresh_account(api_url))


def delete_refresh_token(api_url: str) -> None:
    try:
        keyring.delete_password(_SERVICE, _refresh_account(api_url))
    except PasswordDeleteError:
        pass


def _refresh_account(api_url: str) -> str:
    return f"{_ACCOUNT_REFRESH}:{api_url.rstrip('/')}"
