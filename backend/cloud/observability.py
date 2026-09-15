"""
The logs the cloud processes write.

In Azure, every line a container prints becomes a record in Log Analytics, and
one JSON object per line is what lets a query filter by request, user or job
without a regex. Locally the same records print as plain text, which reads
better in a terminal. `INTERROAI_LOG_FORMAT` picks the format.

Three context variables carry what a line should say about where it came from:
the request (set by `cloud.api.middleware`), the signed-in user (set by
`cloud.api.deps.current_user`) and the index job the worker is running. A filter
attaches them to every record, so no call site has to pass them along.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)
job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)

_CONTEXT = {"request_id": request_id, "user_id": user_id, "job_id": job_id}
#: Attributes every LogRecord has, so they are not repeated as extra fields.
_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", *_CONTEXT}


class ContextFilter(logging.Filter):
    """Copy the context variables onto each record, unless the call site set them."""

    def filter(self, record: logging.LogRecord) -> bool:
        for name, variable in _CONTEXT.items():
            if getattr(record, name, None) is None:
                setattr(record, name, variable.get())
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in _CONTEXT:
            value = getattr(record, name, None)
            if value is not None:
                entry[name] = value
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(*, log_format: str = "text", level: str = "INFO") -> None:
    """Send every logger, uvicorn's included, to stdout in *log_format*."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Uvicorn installs its own handlers; route its records through the root instead,
    # so they come out in the same format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
