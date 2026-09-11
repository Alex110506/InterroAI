"""
User settings for the CLI — the `/user` command's data layer.

Two values, deliberately stored in two different places:

  * the user name goes to ``~/.interroai/config.json`` (not sensitive)
  * the OpenAI API key goes to the OS keychain via ``core.security``

The key is never written to the config file, never echoed back, and never
logged. `masked_api_key` exists so the menu can show that a key is present
without revealing it.
"""
from __future__ import annotations

from config import app_config
from core.security import retrieve_openai_key, store_openai_key

_NAME_KEY = "user_name"
_MAX_NAME_LENGTH = 100


def get_user_name() -> str:
    return app_config.get(_NAME_KEY, "") or ""


def set_user_name(name: str) -> None:
    app_config.set(_NAME_KEY, name.strip()[:_MAX_NAME_LENGTH])


def has_api_key() -> bool:
    return bool(retrieve_openai_key())


def set_api_key(key: str) -> None:
    store_openai_key(key.strip())


def masked_api_key() -> str:
    """
    A recognisable but non-recoverable rendering of the stored key.

    Shows only the last four characters — enough to tell two keys apart when
    checking which one is configured, useless to anyone reading over your
    shoulder or scrolling back through the terminal.
    """
    key = retrieve_openai_key()
    if not key:
        return ""
    tail = key[-4:] if len(key) > 4 else ""
    return f"{'•' * 8}{tail}"
