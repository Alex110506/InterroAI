"""
Secure credential storage using the OS native keychain via keyring.

One credential lives here: the InterroAI cloud's refresh token. No model key is
stored on this machine — the platform holds it, behind the LLM gateway.
"""

import keyring
from keyring.errors import PasswordDeleteError

_SERVICE = "interroai"


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
