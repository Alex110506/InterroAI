"""
Chat session — the agent pipeline, independent of any transport.

One user message flows through two steps:

  Step 1 — Intent classification (2-way):
    A fast model classifies the user's request as one of:
      • "answer"    — general question; reply directly with Markdown
      • "implement" — implementation task; go straight to the coder

  Step 2 — Model selection:
    The caller picks the model explicitly; we validate the ID against the
    coder's model map and reject anything unknown.

  Step 3 — Route:
    • answer:    Q&A agent, then finish
    • implement: coder directly

`ChatSession` owns that flow and emits plain dict events; `api/chat.py` relays
them to the Electron app as WebSocket frames. The WebSocket never appears below
this line — which is the point: the pipeline had previously grown inside a
WebSocket handler and could not be driven any other way.

Model calls go through a `ModelGateway` (`core/models/gateway.py`), handed in or chosen
by `core.providers`. The session resolves it once and passes the same one to
the classifier and the coder, so a request never talks to two different
backends halfway through.

A session is single-request and disposable, but not amnesiac: the caller hands
it the conversation so far as `history`, so a follow-up like "sure" or "now do
the same for the other file" resolves against what was already said. Nothing is
persisted — the transcript lives in the caller's memory and dies with it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from agents import supervisor
from agents.coder import _MODEL_MAP as _CODER_MODEL_MAP
from core import providers
from core.errors import InterroAIError
from core.models.gateway import FAST_TIMEOUT, ModelGateway

logger = logging.getLogger(__name__)

# Used when the caller omits a model entirely. Any value that *is* supplied
# must be a known display ID — we reject unknown ones rather than silently
# substituting a different (possibly pricier) model.
_DEFAULT_MODEL = "gpt-5.6-sol"

#: Display IDs a caller may choose from, best-first. `test_session.py` holds
#: this equal to the coder's model map, so validation cannot drift from what the
#: coder resolves. (The Electron picker keeps its own copy, `MODELS` in
#: `frontend/src/components/ChatPanel.jsx`, which nothing checks.)
AVAILABLE_MODELS: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)

#: How hard the model should think, chosen per request and independent of the
#: model. Passed straight through as OpenAI's `reasoning_effort`, so these are
#: its values, not names of our own.
AVAILABLE_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: Used when the caller names no effort. OpenAI's own default, and the one that
#: does not quietly make every request expensive.
_DEFAULT_EFFORT = "medium"

#: What the intent classifier runs on. Routing is a small prompt that must feel
#: instant, so it takes the least effort the family offers — and a model of its
#: own, so changing the picker's default never changes the cost of routing.
_INTENT_MODEL = "gpt-5.6-luna"
_INTENT_EFFORT = "low"

_INTENT_SYSTEM = """\
You are a routing agent for an AI coding assistant. Classify the user's request into exactly one of two actions:

1. "answer" — The user is asking a general question, requesting an explanation, or having a conversation.
   They are NOT asking to build, change, fix, or implement anything in the codebase.

2. "implement" — The user wants to implement/build/fix/modify something in the codebase.

A reply may be short and depend entirely on the conversation so far ("sure", "yes, do that",
"the second one"). Read it in the context of the earlier turns and classify what the user is
actually asking for, not the literal words in isolation.

Respond strictly with JSON: {"action": "answer"} or {"action": "implement"}
"""

#: How much prior conversation to carry, in characters, newest first. Context
#: is what makes "sure" mean something; an unbounded transcript is what makes
#: the twentieth request cost twenty times the first. This bounds the trade.
_MAX_HISTORY_CHARS = 12_000


def trim_history(history: list[dict] | None) -> list[dict]:
    """Return the most recent turns that fit inside `_MAX_HISTORY_CHARS`."""
    if not history:
        return []

    kept: list[dict] = []
    budget = _MAX_HISTORY_CHARS
    for turn in reversed(history):
        cost = len(turn.get("content") or "")
        if cost > budget:
            break
        budget -= cost
        kept.append(turn)
    kept.reverse()
    return kept

_VALID_ACTIONS = ("answer", "implement")


def is_known_model(model: str) -> bool:
    """True when *model* is a display ID the coder can resolve."""
    return model in _CODER_MODEL_MAP


def unknown_model_message(model: str) -> str:
    return (
        f"Unknown model {model!r}. "
        f"Expected one of: {', '.join(sorted(_CODER_MODEL_MAP))}."
    )


def is_known_effort(effort: str) -> bool:
    """True when *effort* is one of the values OpenAI accepts."""
    return effort in AVAILABLE_EFFORTS


def unknown_effort_message(effort: str) -> str:
    return (
        f"Unknown effort {effort!r}. "
        f"Expected one of: {', '.join(AVAILABLE_EFFORTS)}."
    )


def _fmt_tree(node: dict, depth: int = 0, max_depth: int = 4) -> str:
    if depth > max_depth or not node:
        return ""
    indent = "  " * depth
    if node.get("kind") == "file":
        return f"{indent}{node['name']}\n"
    lines = [f"{indent}{node.get('name', '?')}/\n"]
    for child in node.get("children") or []:
        lines.append(_fmt_tree(child, depth + 1, max_depth))
    return "".join(lines)


def _fmt_git(git: dict) -> str:
    if not git or not git.get("is_git_repo"):
        return "Not a git repository."
    parts: list[str] = []
    if git.get("branch"):
        parts.append(f"Branch: {git['branch']}")
    if git.get("modified_files"):
        parts.append("Modified:\n" + "\n".join(f"  {f}" for f in git["modified_files"]))
    if git.get("recent_commits"):
        parts.append("Recent commits:\n" + "\n".join(f"  {c}" for c in git["recent_commits"]))
    return "\n".join(parts) or "Clean working tree."


async def classify_intent(
    user_message: str,
    project_index: dict,
    history: list[dict] | None = None,
    *,
    gateway: ModelGateway | None = None,
) -> str:
    """
    2-way intent classifier. Returns: 'answer' | 'implement'.

    *history* is the conversation so far. Without it a reply like "sure" has no
    meaning to classify, and the router guesses.

    Only *unusable model output* falls back to 'implement'. A missing API key or
    an unreachable provider propagates: those are the user's real problem, and
    routing past them just produces a second, more confusing failure downstream.
    """
    tree_str = _fmt_tree(project_index.get("file_tree") or {})
    git_str = _fmt_git(project_index.get("git_context") or {})
    context = f"PROJECT STRUCTURE:\n{tree_str or '(empty)'}\n\nGIT CONTEXT:\n{git_str}"

    gateway = gateway or providers.model_gateway()
    # No `temperature`: every model the app offers is a reasoning model, and
    # they refuse it outright. Determinism comes from the prompt and JSON mode
    # instead, and `reasoning_effort` keeps routing cheap.
    response = await gateway.chat(
        timeout=FAST_TIMEOUT,
        model=_INTENT_MODEL,
        messages=[
            {"role": "system", "content": f"{_INTENT_SYSTEM}\n\n{context}"},
            *trim_history(history),
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        reasoning_effort=_INTENT_EFFORT,
    )

    raw = response.choices[0].message.content
    try:
        action = json.loads(raw or "").get("action")
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Intent classifier returned unparseable JSON (%r) — defaulting to 'implement'.",
            raw,
        )
        return "implement"

    if action not in _VALID_ACTIONS:
        logger.warning(
            "Intent classifier returned unknown action %r — defaulting to 'implement'.",
            action,
        )
        return "implement"
    return action


class ChatSession:
    """
    One user request, from classification through to a finished implementation.

    Usage:

        session = ChatSession(project_path, project_index, model, history)
        async for event in session.start("add retries to the uploader"):
            render(event)

    `finished` is True once the work is done and the session should be thrown
    away. It stays False only when `start()` rejected the request outright
    (an unknown model or effort) — the caller can pick a valid one and try
    again on the same session.
    """

    def __init__(
        self,
        project_path: str,
        project_index: dict | None = None,
        model: str | None = None,
        history: list[dict] | None = None,
        gateway: ModelGateway | None = None,
        effort: str | None = None,
    ) -> None:
        self.project_path = project_path
        self.project_index = project_index or {}
        self.model = model or _DEFAULT_MODEL
        #: Chosen per request, independently of the model.
        self.effort = effort or _DEFAULT_EFFORT
        self.history = trim_history(history)
        self.gateway = gateway or providers.model_gateway()
        self.finished = False

    # ── Public entry point ────────────────────────────────────────────────

    async def start(self, user_message: str) -> AsyncIterator[dict]:
        """Classify *user_message* and route it. Yields the resulting events."""
        if not is_known_model(self.model):
            # Deliberately non-terminal: the caller can pick a valid model and
            # carry on with the same session rather than starting over.
            yield {"type": "error", "message": unknown_model_message(self.model)}
            return
        if not is_known_effort(self.effort):
            yield {"type": "error", "message": unknown_effort_message(self.effort)}
            return

        logger.info("User selected model: %s (effort: %s)", self.model, self.effort)

        async for event in self._guard(self._route(user_message)):
            yield event

    # ── Routing ───────────────────────────────────────────────────────────

    async def _route(self, user_message: str) -> AsyncIterator[dict]:
        action = await classify_intent(
            user_message, self.project_index, self.history, gateway=self.gateway
        )
        logger.info("Intent classified: %r", action)

        yield {"type": "ready", "refined_prompt": user_message}
        logger.info(
            "Dispatching to %s | project=%r | model=%r",
            "Q&A agent" if action == "answer" else "coder",
            self.project_path,
            self.model,
        )
        async for event in self._stream_agent(user_message, intent=action):
            yield event

    async def _stream_agent(self, prompt: str, *, intent: str) -> AsyncIterator[dict]:
        async for event in supervisor.stream(
            prompt,
            self.project_path,
            self.model,
            intent=intent,
            history=self.history,
            gateway=self.gateway,
            effort=self.effort,
        ):
            yield event
        self.finished = True

    # ── Error handling ────────────────────────────────────────────────────

    async def _guard(self, stream: AsyncIterator[dict]) -> AsyncIterator[dict]:
        """
        Turn a failure into a final `error` event.

        Both transports have to render errors for the user regardless, and
        neither can recover the session once the pipeline has thrown — so the
        failure is reported here, once, instead of in every caller.
        """
        try:
            async for event in stream:
                yield event
        except InterroAIError as exc:
            # Expected and already phrased for a human (missing key, bad model,
            # signed out of the cloud). A `code`, when there is one, lets the app
            # act on it rather than only show it.
            logger.info("Chat session ended: %s", exc)
            self.finished = True
            event = {"type": "error", "message": str(exc)}
            if exc.code:
                event["code"] = exc.code
            yield event
        except Exception as exc:  # noqa: BLE001
            # Unexpected: keep the traceback. The caller still gets the message,
            # since this is a local single-user tool and the alternative is a
            # dead prompt with no explanation.
            logger.exception("Unhandled error in chat session")
            self.finished = True
            yield {"type": "error", "message": f"Internal error: {exc}"}
