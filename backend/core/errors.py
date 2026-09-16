"""
Typed error hierarchy.

The point of this module is to let call sites distinguish two very different
kinds of failure that used to be flattened into the same `except Exception`:

  * **Expected** — the user has no API key, the model referenced a file that
    doesn't exist, a search block didn't match. These are normal operating
    conditions. They carry a message that is safe (and useful) to show to the
    user or hand back to the model, and they should NOT produce a stack trace.

  * **Unexpected** — anything else. A bug. These must keep their traceback and
    be logged with `logger.exception`, never silently stringified.

Anything raised deliberately by InterroAI inherits from `InterroAIError`, so a
handler can catch the expected cases precisely and let real defects surface.
"""
from __future__ import annotations


class InterroAIError(Exception):
    """Base class for every error InterroAI raises deliberately."""

    #: A stable name the app can act on, sent beside the message in error events.
    #: None for errors that are only ever shown.
    code: str | None = None


class MissingAPIKeyError(InterroAIError):
    """No OpenAI API key is stored in the OS keychain."""

    def __init__(
        self,
        message: str = "No OpenAI API key configured — add yours in Settings.",
    ) -> None:
        super().__init__(message)


class UpstreamLLMError(InterroAIError):
    """
    An OpenAI request failed and could not be recovered.

    Raised after the retry policy in `core.models.llm` has exhausted its attempts, so
    reaching this means the provider was genuinely unavailable rather than
    momentarily flaky.
    """


# ── Tool failures ────────────────────────────────────────────────────────────
# These are raised by the Coder Agent's tools. The agent catches `ToolError`
# and feeds `str(exc)` back to the model as a tool result, so the message must
# read as actionable instruction to the model — it is how the model learns to
# correct itself.


class ToolError(InterroAIError):
    """An agent tool failed in a way the *model* can recover from."""


class PathEscapeError(ToolError):
    """The model asked for a path outside the project root."""


class FileNotFoundInProjectError(ToolError):
    """The model referenced a file that does not exist in the project."""


class PatchError(ToolError):
    """A patch could not be applied (search block missing or ambiguous)."""


# ── The cloud ────────────────────────────────────────────────────────────────
# Raised by the Cloud API clients in `core/remote/`. Each carries a `code`, so
# the app can react to it (show the sign-in screen, say the day's allowance is
# used up) without matching on message text.


class CloudError(InterroAIError):
    """The InterroAI cloud refused or failed a request."""

    code = "cloud_error"
    default_message = "The InterroAI cloud refused the request."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message or self.default_message)
        if code:
            self.code = code
        #: The API's error detail as sent, e.g. the running job's id beside
        #: `job_already_running`.
        self.details = details or {}


class NotSignedInError(CloudError):
    code = "not_signed_in"
    default_message = "Sign in to InterroAI to continue."


class QuotaExceededError(CloudError):
    code = "quota_exceeded"
    default_message = "Today's InterroAI allowance is used up. It resets at midnight UTC."


class CloudUnavailableError(CloudError):
    code = "cloud_unavailable"
    default_message = "The InterroAI cloud can't be reached right now."
