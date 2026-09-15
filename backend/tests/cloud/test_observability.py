"""The cloud processes' logs: JSON lines carrying request, user and job ids."""
from __future__ import annotations

import json
import logging
import sys

import pytest

from cloud import observability
from cloud.observability import ContextFilter, JsonFormatter, configure_logging


def _record(message: str = "hello %s", args: tuple = ("world",), **fields) -> logging.LogRecord:
    record = logging.makeLogRecord(
        {
            "name": "cloud.test",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": message,
            "args": args,
            **fields,
        }
    )
    ContextFilter().filter(record)
    return record


def _json(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_a_line_carries_the_message_its_level_and_a_utc_time():
    entry = _json(_record())

    assert entry["message"] == "hello world"
    assert (entry["level"], entry["logger"]) == ("INFO", "cloud.test")
    assert entry["time"].endswith("+00:00")


def test_the_ids_in_context_are_attached_to_every_line():
    request = observability.request_id.set("req-12345678")
    job = observability.job_id.set("job-9")
    try:
        entry = _json(_record())
    finally:
        observability.job_id.reset(job)
        observability.request_id.reset(request)

    assert (entry["request_id"], entry["job_id"]) == ("req-12345678", "job-9")
    assert "user_id" not in entry, "an id nobody set is left out, not written as null"


def test_extra_fields_become_json_fields():
    entry = _json(_record(status=404, duration_ms=12.5))
    assert (entry["status"], entry["duration_ms"]) == (404, 12.5)


def test_an_exception_is_written_with_its_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(exc_info=sys.exc_info())

    assert "ValueError: boom" in _json(record)["exception"]


@pytest.fixture
def restored_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


def test_json_logging_prints_one_object_per_line(restored_root_logger, capsys):
    configure_logging(log_format="json", level="INFO")

    logging.getLogger("cloud.test").info("started", extra={"replicas": 2})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line) | {"time": None} == {
        "time": None,
        "level": "INFO",
        "logger": "cloud.test",
        "message": "started",
        "replicas": 2,
    }
