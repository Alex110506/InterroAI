"""Non-sensitive application configuration persisted to ~/.interroai/config.json."""

import json
from pathlib import Path

APP_DIR = Path.home() / ".interroai"
_CONFIG_FILE = APP_DIR / "config.json"


class _AppConfig:
    def __init__(self) -> None:
        APP_DIR.mkdir(exist_ok=True)
        self._data: dict = {}
        if _CONFIG_FILE.exists():
            try:
                self._data = json.loads(_CONFIG_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        _CONFIG_FILE.write_text(json.dumps(self._data, indent=2))

    def all(self) -> dict:
        return dict(self._data)


app_config = _AppConfig()
