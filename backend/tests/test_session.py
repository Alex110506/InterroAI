"""
The chat pipeline itself, driven without any transport.

`ChatSession` is what both the CLI and the WebSocket adapter run, so these
tests pin the routing and the error contract once, at the level where they
actually live. `test_chat.py` covers the WebSocket adapter on top;
`test_pipeline.py` drives the two end to end.
"""
from __future__ import annotations

import json

import pytest
from conftest import make_response

import agents.session as session
from agents.coder import _MODEL_MAP
from agents.session import ChatSession
from core.errors import MissingAPIKeyError


def _queue_intent(monkeypatch, content):
    async def fake_completion(client, **kwargs):
        return make_response(content=content)

    monkeypatch.setattr(session, "get_client", lambda timeout: object())
    monkeypatch.setattr(session, "chat_completion", fake_completion)


# ── Intent classification ────────────────────────────────────────────────────


@pytest.mark.parametrize("action", ["answer", "implement"])
async def test_each_valid_action_is_returned(monkeypatch, action):
    _queue_intent(monkeypatch, json.dumps({"action": action}))
    assert await session.classify_intent("do a thing", {}) == action


async def test_an_unknown_action_falls_back_to_implement(monkeypatch, caplog):
    _queue_intent(monkeypatch, json.dumps({"action": "delete_production"}))
    with caplog.at_level("WARNING", logger="agents.session"):
        assert await session.classify_intent("x", {}) == "implement"
    assert any("unknown action" in r.message for r in caplog.records)


async def test_malformed_json_falls_back_to_implement(monkeypatch, caplog):
    """Recoverable — but it must be logged, not silent."""
    _queue_intent(monkeypatch, "this is not json")
    with caplog.at_level("WARNING", logger="agents.session"):
        assert await session.classify_intent("x", {}) == "implement"
    assert any("unparseable" in r.message for r in caplog.records)


async def test_empty_content_falls_back_to_implement(monkeypatch):
    _queue_intent(monkeypatch, None)
    assert await session.classify_intent("x", {}) == "implement"


async def test_json_without_an_action_key_falls_back(monkeypatch):
    _queue_intent(monkeypatch, json.dumps({"something_else": 1}))
    assert await session.classify_intent("x", {}) == "implement"


async def test_a_missing_key_propagates_instead_of_misrouting(monkeypatch):
    """
    The regression: this used to return "implement", sending the user to the
    coder, which then failed again for an apparently unrelated reason.
    """
    def boom(timeout):
        raise MissingAPIKeyError()

    monkeypatch.setattr(session, "get_client", boom)
    with pytest.raises(MissingAPIKeyError):
        await session.classify_intent("x", {})


async def test_provider_errors_propagate(monkeypatch):
    async def boom(client, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(session, "get_client", lambda timeout: object())
    monkeypatch.setattr(session, "chat_completion", boom)
    with pytest.raises(RuntimeError):
        await session.classify_intent("x", {})


async def test_project_context_is_included_in_the_prompt(monkeypatch):
    captured = {}

    async def capture(client, **kwargs):
        captured.update(kwargs)
        return make_response(content=json.dumps({"action": "answer"}))

    monkeypatch.setattr(session, "get_client", lambda timeout: object())
    monkeypatch.setattr(session, "chat_completion", capture)

    index = {
        "file_tree": {"kind": "dir", "name": "src", "children": []},
        "git_context": {"is_git_repo": True, "branch": "main"},
    }
    await session.classify_intent("what does this do?", index)

    system = captured["messages"][0]["content"]
    assert "src/" in system
    assert "main" in system


async def test_classification_is_deterministic(monkeypatch):
    """Routing must not wobble between runs for the same request."""
    captured = {}

    async def capture(client, **kwargs):
        captured.update(kwargs)
        return make_response(content=json.dumps({"action": "answer"}))

    monkeypatch.setattr(session, "get_client", lambda timeout: object())
    monkeypatch.setattr(session, "chat_completion", capture)
    await session.classify_intent("x", {})
    assert captured["temperature"] == 0.0


# ── The advertised model list ────────────────────────────────────────────────


def test_the_advertised_models_are_exactly_the_resolvable_ones():
    """
    `AVAILABLE_MODELS` is what `/model` offers and `_MODEL_MAP` is what the
    coder can resolve. When the UI lived in JSX these two drifted silently and
    the backend rejected an ID its own picker had offered.
    """
    assert set(session.AVAILABLE_MODELS) == set(_MODEL_MAP)


def test_the_default_model_is_offered_by_the_picker():
    assert session._DEFAULT_MODEL in session.AVAILABLE_MODELS


# ── Session routing ──────────────────────────────────────────────────────────


@pytest.fixture
def captured_agent(monkeypatch):
    """Replace the supervisor so no agent or network work actually happens."""
    calls: list[dict] = []

    async def fake_stream(prompt, project_path, model, intent="implement", history=None):
        calls.append(
            {
                "prompt": prompt,
                "project_path": project_path,
                "model": model,
                "intent": intent,
                "history": list(history or []),
            }
        )
        yield {"type": "done", "summary": "stubbed"}

    monkeypatch.setattr(session.supervisor, "stream", fake_stream)
    return calls


def _fix_intent(monkeypatch, action):
    async def fake_classify(message, index, history=None):
        return action

    monkeypatch.setattr(session, "classify_intent", fake_classify)


async def _drain(stream) -> list[dict]:
    return [event async for event in stream]


async def test_an_unknown_model_is_rejected_before_any_api_call(monkeypatch):
    def explode(timeout):
        raise AssertionError("the model must be validated before the provider is touched")

    monkeypatch.setattr(session, "get_client", explode)
    chat = ChatSession("/tmp/p", {}, model="gpt-9-ultra")
    events = await _drain(chat.start("hi"))

    assert events[0]["type"] == "error"
    assert "Unknown model" in events[0]["message"]
    assert "gpt-5.4-mini" in events[0]["message"], "the error should list valid ids"


async def test_a_rejected_model_leaves_the_session_usable(monkeypatch):
    chat = ChatSession("/tmp/p", {}, model="bogus")
    await _drain(chat.start("hi"))
    assert chat.finished is False


async def test_omitting_the_model_uses_the_default(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {})
    await _drain(chat.start("rename X"))
    assert captured_agent[0]["model"] == session._DEFAULT_MODEL


async def test_an_answer_goes_straight_to_the_qa_agent(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "answer")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")
    events = await _drain(chat.start("what does this repo do?"))

    assert events[0]["type"] == "ready"
    assert captured_agent[0]["intent"] == "answer"
    assert chat.finished is True


async def test_a_clear_request_goes_straight_to_the_coder(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")
    events = await _drain(chat.start("rename X to Y in a.py"))

    assert events[0]["type"] == "ready"
    assert captured_agent[0]["intent"] == "implement"
    assert captured_agent[0]["prompt"] == "rename X to Y in a.py"
    assert chat.finished is True


# ── Error contract ───────────────────────────────────────────────────────────


async def test_a_missing_key_becomes_a_final_error_event(monkeypatch):
    """
    Both transports have to render this, so the session converts it once
    rather than leaving each caller to catch it.
    """
    def boom(timeout):
        raise MissingAPIKeyError()

    monkeypatch.setattr(session, "get_client", boom)
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")
    events = await _drain(chat.start("hello"))

    assert events[-1]["type"] == "error"
    assert "API key" in events[-1]["message"]
    assert chat.finished is True, "an unrecoverable failure must end the session"


async def test_an_unexpected_failure_is_reported_and_logged(monkeypatch, caplog):
    async def boom(message, index, history=None):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(session, "classify_intent", boom)
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")

    with caplog.at_level("ERROR", logger="agents.session"):
        events = await _drain(chat.start("hello"))

    assert events[-1]["type"] == "error"
    assert "classifier exploded" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records), "the traceback must be kept"
    assert chat.finished is True


async def test_a_new_session_is_independent_of_earlier_ones(monkeypatch, captured_agent):
    """Sessions are single-use and disposable, whatever conversation they were handed."""
    _fix_intent(monkeypatch, "implement")

    first = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")
    await _drain(first.start("first request"))
    second = ChatSession(
        "/tmp/p", {}, model="gpt-5.4-mini",
        history=[{"role": "user", "content": "first request"}],
    )
    await _drain(second.start("second request"))

    assert [c["prompt"] for c in captured_agent] == ["first request", "second request"]
    assert first.finished is True
    assert second.finished is True


# ── Conversation context ─────────────────────────────────────────────────────


CONVERSATION = [
    {"role": "user", "content": "what is this project about"},
    {"role": "assistant", "content": "A social chat and calling app."},
]


async def test_the_conversation_reaches_the_classifier(monkeypatch):
    """
    The reported bug: "sure" was classified with no idea what it agreed to,
    because the classifier only ever saw the latest message.
    """
    captured = {}

    async def capture(client, **kwargs):
        captured.update(kwargs)
        return make_response(content=json.dumps({"action": "answer"}))

    monkeypatch.setattr(session, "get_client", lambda timeout: object())
    monkeypatch.setattr(session, "chat_completion", capture)

    await session.classify_intent("sure", {}, CONVERSATION)

    contents = [m["content"] for m in captured["messages"]]
    assert "A social chat and calling app." in contents
    assert contents[-1] == "sure", "the new message must still come last"


async def test_the_conversation_reaches_the_agent(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "answer")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini", history=CONVERSATION)
    await _drain(chat.start("sure"))

    assert captured_agent[0]["history"] == CONVERSATION


async def test_a_session_with_no_history_still_works(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini")
    await _drain(chat.start("do it"))

    assert chat.history == []
    assert captured_agent[0]["history"] == []


# ── History trimming ─────────────────────────────────────────────────────────


def test_a_short_conversation_is_kept_whole():
    assert session.trim_history(CONVERSATION) == CONVERSATION


def test_no_history_trims_to_nothing():
    assert session.trim_history(None) == []
    assert session.trim_history([]) == []


def test_an_oversized_conversation_keeps_the_newest_turns():
    """
    Context is what makes a follow-up work; an unbounded transcript is what
    makes the twentieth request cost twenty times the first.
    """
    turns = [
        {"role": "user", "content": f"{i}" + "x" * 5_000}
        for i in range(10)
    ]
    kept = session.trim_history(turns)

    assert kept, "something must survive"
    assert len(kept) < len(turns), "an oversized conversation must be trimmed"
    assert kept[-1] is turns[-1], "the most recent turn is the one that matters"
    assert sum(len(t["content"]) for t in kept) <= session._MAX_HISTORY_CHARS


def test_a_single_oversized_turn_does_not_wedge_the_conversation():
    """One enormous paste must not silently drop every later turn with it."""
    turns = [
        {"role": "user", "content": "y" * (session._MAX_HISTORY_CHARS + 1)},
        {"role": "assistant", "content": "short reply"},
    ]
    assert session.trim_history(turns) == [turns[1]]


def test_the_session_trims_what_it_is_given():
    turns = [{"role": "user", "content": "z" * 9_000} for _ in range(3)]
    chat = ChatSession("/tmp/p", {}, model="gpt-5.4-mini", history=turns)
    assert len(chat.history) < len(turns)
