"""
The exception hierarchy encodes a policy, so these tests assert the
relationships rather than the classes themselves.

The critical invariant is the *negative* one: `MissingAPIKeyError` must not be a
`ToolError`. `CoderAgent._run_tool` catches `ToolError` and hands the message
back to the model to retry; if a missing key were caught there, the agent would
burn its tool rounds "retrying" something no model can fix.
"""
from __future__ import annotations

import pytest

from core.errors import (
    FileNotFoundInProjectError,
    InterroAIError,
    MissingAPIKeyError,
    PatchError,
    PathEscapeError,
    ToolError,
    UpstreamLLMError,
)

ALL_ERRORS = [
    MissingAPIKeyError,
    UpstreamLLMError,
    ToolError,
    PathEscapeError,
    FileNotFoundInProjectError,
    PatchError,
]

TOOL_ERRORS = [ToolError, PathEscapeError, FileNotFoundInProjectError, PatchError]


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_everything_inherits_from_the_base(error_cls):
    """One `except InterroAIError` must be able to catch any deliberate failure."""
    assert issubclass(error_cls, InterroAIError)
    assert issubclass(error_cls, Exception)


@pytest.mark.parametrize("error_cls", TOOL_ERRORS)
def test_tool_errors_are_catchable_as_tool_error(error_cls):
    assert issubclass(error_cls, ToolError)
    with pytest.raises(ToolError):
        raise error_cls("boom")


@pytest.mark.parametrize("error_cls", [MissingAPIKeyError, UpstreamLLMError])
def test_terminal_errors_are_not_tool_errors(error_cls):
    """These must escape `except ToolError` so they terminate the run."""
    assert not issubclass(error_cls, ToolError)


def test_missing_api_key_has_actionable_default_message():
    """The message is shown to the user verbatim, so it must name the fix."""
    message = str(MissingAPIKeyError())
    assert "API key" in message
    assert "Settings" in message


def test_missing_api_key_accepts_a_custom_message():
    assert str(MissingAPIKeyError("custom text")) == "custom text"


def test_tool_error_message_round_trips():
    """`_run_tool` returns `str(exc)` to the model — nothing may be swallowed."""
    assert str(PatchError("search block appears 3 times")) == "search block appears 3 times"


def test_base_error_is_not_caught_as_builtin_value_error():
    """Guards the migration away from the old bare `raise ValueError(...)`."""
    assert not issubclass(InterroAIError, ValueError)
