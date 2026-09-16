"""
The chat pipeline itself, driven without any transport.

`ChatSession` is what the WebSocket adapter runs, so these tests pin the
routing and the error contract once, at the level where they actually live.
`test_chat.py` covers the WebSocket adapter on top; `test_pipeline.py` drives
the two end to end.

Model calls are answered by a `FakeGateway` handed straight to the code under
test, so nothing here patches a module global to keep the network away.
"""
from __future__ import annotations

import json

import pytest
from conftest import FakeGateway, make_response

import agents.session as session
from agents.coder import _MODEL_MAP
from agents.session import ChatSession
from core.errors import MissingAPIKeyError
from core.models.gateway import FAST_TIMEOUT


def _intent_gateway(content):
    """A gateway whose one reply is the classifier's raw output."""
    return FakeGateway([make_response(content=content)])


# ── Intent classification ────────────────────────────────────────────────────


@pytest.mark.parametrize("action", ["answer", "implement"])
async def test_each_valid_action_is_returned(action):
    gateway = _intent_gateway(json.dumps({"action": action}))
    assert await session.classify_intent("do a thing", {}, gateway=gateway) == action


async def test_an_unknown_action_falls_back_to_implement(caplog):
    gateway = _intent_gateway(json.dumps({"action": "delete_production"}))
    with caplog.at_level("WARNING", logger="agents.session"):
        assert await session.classify_intent("x", {}, gateway=gateway) == "implement"
    assert any("unknown action" in r.message for r in caplog.records)


async def test_malformed_json_falls_back_to_implement(caplog):
    """Recoverable — but it must be logged, not silent."""
    gateway = _intent_gateway("this is not json")
    with caplog.at_level("WARNING", logger="agents.session"):
        assert await session.classify_intent("x", {}, gateway=gateway) == "implement"
    assert any("unparseable" in r.message for r in caplog.records)


async def test_empty_content_falls_back_to_implement():
    gateway = _intent_gateway(None)
    assert await session.classify_intent("x", {}, gateway=gateway) == "implement"


async def test_json_without_an_action_key_falls_back():
    gateway = _intent_gateway(json.dumps({"something_else": 1}))
    assert await session.classify_intent("x", {}, gateway=gateway) == "implement"


async def test_a_missing_key_propagates_instead_of_misrouting():
    """
    The regression: this used to return "implement", sending the user to the
    coder, which then failed again for an apparently unrelated reason.
    """
    gateway = FakeGateway([MissingAPIKeyError()])
    with pytest.raises(MissingAPIKeyError):
        await session.classify_intent("x", {}, gateway=gateway)


async def test_provider_errors_propagate():
    gateway = FakeGateway([RuntimeError("provider down")])
    with pytest.raises(RuntimeError):
        await session.classify_intent("x", {}, gateway=gateway)


async def test_project_context_is_included_in_the_prompt():
    gateway = _intent_gateway(json.dumps({"action": "answer"}))
    index = {
        "file_tree": {"kind": "dir", "name": "src", "children": []},
        "git_context": {"is_git_repo": True, "branch": "main"},
    }
    await session.classify_intent("what does this do?", index, gateway=gateway)

    system = gateway.requests[0]["messages"][0]["content"]
    assert "src/" in system
    assert "main" in system


async def test_classification_sends_no_temperature_and_asks_for_little_effort():
    """
    Every model the app offers reasons, and a reasoning model rejects
    `temperature` outright. Routing stays cheap through `reasoning_effort`, and
    on a model of its own so the picker's default never changes its cost.
    """
    gateway = _intent_gateway(json.dumps({"action": "answer"}))
    await session.classify_intent("x", {}, gateway=gateway)

    assert "temperature" not in gateway.requests[0]
    assert gateway.requests[0]["model"] == session._INTENT_MODEL
    assert gateway.requests[0]["reasoning_effort"] == session._INTENT_EFFORT


async def test_classification_uses_the_fast_timeout():
    """Routing is a small prompt; it must not inherit the coder's long budget."""
    gateway = _intent_gateway(json.dumps({"action": "answer"}))
    await session.classify_intent("x", {}, gateway=gateway)
    assert gateway.timeouts == [FAST_TIMEOUT]


async def test_without_a_gateway_the_composition_root_chooses(monkeypatch):
    gateway = _intent_gateway(json.dumps({"action": "answer"}))
    monkeypatch.setattr(session.providers, "model_gateway", lambda: gateway)

    assert await session.classify_intent("x", {}) == "answer"
    assert gateway.requests, "the provider's gateway must have been used"


# ── The advertised model list ────────────────────────────────────────────────


def test_the_advertised_models_are_exactly_the_resolvable_ones():
    """
    `AVAILABLE_MODELS` is what the backend advertises and `_MODEL_MAP` is what
    the coder can resolve. When these two drifted, the backend rejected an ID
    its own picker had offered.
    """
    assert set(session.AVAILABLE_MODELS) == set(_MODEL_MAP)


def test_the_default_model_is_offered_by_the_picker():
    assert session._DEFAULT_MODEL in session.AVAILABLE_MODELS


def test_the_offered_efforts_are_openais_own_values():
    """They are passed straight through as `reasoning_effort`, not translated."""
    assert session.AVAILABLE_EFFORTS == ("low", "medium", "high", "xhigh", "max")


def test_the_default_effort_is_one_of_the_offered_ones():
    assert session._DEFAULT_EFFORT in session.AVAILABLE_EFFORTS


def test_the_classifier_runs_on_an_offered_model():
    """Routing must not name a model the gateway's allowlist would refuse."""
    assert session._INTENT_MODEL in session.AVAILABLE_MODELS
    assert session._INTENT_EFFORT in session.AVAILABLE_EFFORTS


# ── Session routing ──────────────────────────────────────────────────────────


@pytest.fixture
def captured_agent(monkeypatch):
    """Replace the supervisor so no agent or network work actually happens."""
    calls: list[dict] = []

    async def fake_stream(
        prompt, project_path, model, intent="implement", history=None, gateway=None, effort=None
    ):
        calls.append(
            {
                "prompt": prompt,
                "project_path": project_path,
                "model": model,
                "intent": intent,
                "history": list(history or []),
                "gateway": gateway,
                "effort": effort,
            }
        )
        yield {"type": "done", "summary": "stubbed"}

    monkeypatch.setattr(session.supervisor, "stream", fake_stream)
    return calls


def _fix_intent(monkeypatch, action):
    async def fake_classify(message, index, history=None, *, gateway=None):
        return action

    monkeypatch.setattr(session, "classify_intent", fake_classify)


async def _drain(stream) -> list[dict]:
    return [event async for event in stream]


async def test_an_unknown_model_is_rejected_before_any_api_call():
    gateway = FakeGateway()
    chat = ChatSession("/tmp/p", {}, model="gpt-9-ultra", gateway=gateway)
    events = await _drain(chat.start("hi"))

    assert events[0]["type"] == "error"
    assert "Unknown model" in events[0]["message"]
    assert "gpt-5.6-sol" in events[0]["message"], "the error should list valid ids"
    assert gateway.requests == [], "the model must be validated before the provider is touched"


async def test_a_rejected_model_leaves_the_session_usable():
    chat = ChatSession("/tmp/p", {}, model="bogus", gateway=FakeGateway())
    await _drain(chat.start("hi"))
    assert chat.finished is False


async def test_omitting_the_model_uses_the_default(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {})
    await _drain(chat.start("rename X"))
    assert captured_agent[0]["model"] == session._DEFAULT_MODEL


async def test_an_unknown_effort_is_rejected_before_any_api_call():
    gateway = FakeGateway()
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", gateway=gateway, effort="ludicrous")
    events = await _drain(chat.start("hi"))

    assert events[0]["type"] == "error"
    assert "Unknown effort" in events[0]["message"]
    assert gateway.requests == [], "the effort must be checked before the provider is touched"


async def test_a_rejected_effort_leaves_the_session_usable():
    chat = ChatSession("/tmp/p", {}, effort="bogus", gateway=FakeGateway())
    await _drain(chat.start("hi"))
    assert chat.finished is False


async def test_the_chosen_effort_reaches_the_agent(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", effort="max")
    await _drain(chat.start("rename X"))
    assert captured_agent[0]["effort"] == "max"


async def test_omitting_the_effort_uses_the_default(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {})
    await _drain(chat.start("rename X"))
    assert captured_agent[0]["effort"] == session._DEFAULT_EFFORT


async def test_an_answer_goes_straight_to_the_qa_agent(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "answer")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol")
    events = await _drain(chat.start("what does this repo do?"))

    assert events[0]["type"] == "ready"
    assert captured_agent[0]["intent"] == "answer"
    assert chat.finished is True


async def test_a_clear_request_goes_straight_to_the_coder(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol")
    events = await _drain(chat.start("rename X to Y in a.py"))

    assert events[0]["type"] == "ready"
    assert captured_agent[0]["intent"] == "implement"
    assert captured_agent[0]["prompt"] == "rename X to Y in a.py"
    assert chat.finished is True


async def test_the_classifier_and_the_agent_share_one_gateway(captured_agent):
    """One request must not talk to two different model backends."""
    gateway = _intent_gateway(json.dumps({"action": "implement"}))
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", gateway=gateway)
    await _drain(chat.start("rename X"))

    assert gateway.requests, "the classifier must have used the session's gateway"
    assert captured_agent[0]["gateway"] is gateway


# ── Error contract ───────────────────────────────────────────────────────────


async def test_a_missing_key_becomes_a_final_error_event():
    """
    Every transport has to render this, so the session converts it once
    rather than leaving each caller to catch it.
    """
    gateway = FakeGateway([MissingAPIKeyError()])
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", gateway=gateway)
    events = await _drain(chat.start("hello"))

    assert events[-1]["type"] == "error"
    assert "API key" in events[-1]["message"]
    assert chat.finished is True, "an unrecoverable failure must end the session"


async def test_an_unexpected_failure_is_reported_and_logged(monkeypatch, caplog):
    async def boom(message, index, history=None, **kwargs):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(session, "classify_intent", boom)
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol")

    with caplog.at_level("ERROR", logger="agents.session"):
        events = await _drain(chat.start("hello"))

    assert events[-1]["type"] == "error"
    assert "classifier exploded" in events[-1]["message"]
    assert any(r.exc_info for r in caplog.records), "the traceback must be kept"
    assert chat.finished is True


async def test_a_new_session_is_independent_of_earlier_ones(monkeypatch, captured_agent):
    """Sessions are single-use and disposable, whatever conversation they were handed."""
    _fix_intent(monkeypatch, "implement")

    first = ChatSession("/tmp/p", {}, model="gpt-5.6-sol")
    await _drain(first.start("first request"))
    second = ChatSession(
        "/tmp/p", {}, model="gpt-5.6-sol",
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


async def test_the_conversation_reaches_the_classifier():
    """
    The reported bug: "sure" was classified with no idea what it agreed to,
    because the classifier only ever saw the latest message.
    """
    gateway = _intent_gateway(json.dumps({"action": "answer"}))
    await session.classify_intent("sure", {}, CONVERSATION, gateway=gateway)

    contents = [m["content"] for m in gateway.requests[0]["messages"]]
    assert "A social chat and calling app." in contents
    assert contents[-1] == "sure", "the new message must still come last"


async def test_the_conversation_reaches_the_agent(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "answer")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", history=CONVERSATION)
    await _drain(chat.start("sure"))

    assert captured_agent[0]["history"] == CONVERSATION


async def test_a_session_with_no_history_still_works(monkeypatch, captured_agent):
    _fix_intent(monkeypatch, "implement")
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol")
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
    chat = ChatSession("/tmp/p", {}, model="gpt-5.6-sol", history=turns)
    assert len(chat.history) < len(turns)
