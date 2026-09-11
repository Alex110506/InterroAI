"""
The terminal frontend: label rules, slash commands, event rendering and the
statelessness guarantee.

Everything here runs headless. `PromptSession` is constructed but never read
from — each test either drives a command directly or replaces `InterroCLI.ask`
with scripted answers, so no test needs a tty.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

import cli.indexing as indexing
import cli.main as cli_main
import cli.settings as cli_settings
from agents.session import _DEFAULT_MODEL, AVAILABLE_MODELS
from cli import commands
from cli.app import InterroCLI
from cli.render import EventRenderer
from cli.theme import THEME, model_label


@pytest.fixture(autouse=True)
def clean_config():
    """
    `config.app_config` is a module-level singleton, so a username set by one
    test would otherwise leak into the next and make the suite order-dependent.
    """
    from config import app_config

    previous = app_config.get("user_name", "")
    app_config.set("user_name", "")
    yield
    app_config.set("user_name", previous)


@pytest.fixture
def console():
    """A Console that captures output instead of touching the terminal."""
    return Console(
        theme=THEME, file=io.StringIO(), force_terminal=False, width=200, legacy_windows=False
    )


def text_of(console: Console) -> str:
    return console.file.getvalue()


@pytest.fixture
def app(console, tmp_path):
    return InterroCLI(project_path=str(tmp_path), console=console)


@pytest.fixture
def scripted(monkeypatch):
    """Feed `app.ask` a queue of answers; records the prompts it was shown."""
    answers: list[str] = []
    prompts: list[str] = []

    async def fake_ask(self, message, *, is_password=False):
        prompts.append(f"{'[password] ' if is_password else ''}{message}")
        return answers.pop(0) if answers else ""

    monkeypatch.setattr(InterroCLI, "ask", fake_ask)
    return answers, prompts


# ── Model labels ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("display_id", "expected"),
    [
        ("gpt-5.5-high-effort", "GPT-5.5 High Effort"),
        ("gpt-5.4-low-effort", "GPT-5.4 Low Effort"),
        ("gpt-5.4-mini", "GPT-5.4 Mini"),
    ],
)
def test_a_display_id_becomes_a_readable_label(display_id, expected):
    assert model_label(display_id) == expected


def test_every_available_model_gets_a_label():
    for model in AVAILABLE_MODELS:
        assert model_label(model)[0].isupper()


# ── /model ───────────────────────────────────────────────────────────────────


async def test_model_lists_every_choice_and_marks_the_current_one(app, scripted):
    await commands.dispatch(app, "/model")
    out = text_of(app.console)

    for i, model in enumerate(AVAILABLE_MODELS, start=1):
        assert f"{i}." in out
        assert model_label(model) in out
    assert "current" in out


async def test_model_selects_by_number(app, scripted):
    answers, _ = scripted
    answers.append("5")
    await commands.dispatch(app, "/model")
    assert app.model == AVAILABLE_MODELS[4]


async def test_model_accepts_an_id_as_an_argument(app):
    await commands.dispatch(app, "/model gpt-5.4-mini")
    assert app.model == "gpt-5.4-mini"


async def test_model_rejects_an_unknown_id(app):
    await commands.dispatch(app, "/model gpt-9-ultra")
    assert app.model == _DEFAULT_MODEL
    assert "Unknown model" in text_of(app.console)


async def test_model_rejects_an_out_of_range_number(app, scripted):
    answers, _ = scripted
    answers.append("99")
    await commands.dispatch(app, "/model")
    assert app.model == _DEFAULT_MODEL
    assert "Not a valid choice" in text_of(app.console)


async def test_an_empty_answer_keeps_the_current_model(app, scripted):
    answers, _ = scripted
    answers.append("")
    await commands.dispatch(app, "/model")
    assert app.model == _DEFAULT_MODEL


# ── /user ────────────────────────────────────────────────────────────────────


async def test_user_offers_exactly_two_numbered_options(app, scripted, fake_keyring):
    await commands.dispatch(app, "/user")
    out = text_of(app.console)

    assert "1." in out and "Username" in out
    assert "2." in out and "OpenAI API key" in out
    assert "3." not in out


async def test_user_option_one_sets_the_username(app, scripted, fake_keyring):
    answers, _ = scripted
    answers.extend(["1", "Alex"])
    await commands.dispatch(app, "/user")
    assert cli_settings.get_user_name() == "Alex"


async def test_user_option_two_stores_the_key_in_the_keychain(app, scripted, fake_keyring):
    answers, _ = scripted
    answers.extend(["2", "sk-secret-value"])
    await commands.dispatch(app, "/user")
    assert fake_keyring[("interroai", "openai_api_key")] == "sk-secret-value"


async def test_the_api_key_is_read_with_masked_input(app, scripted, fake_keyring):
    """It must not be recoverable from the terminal scrollback."""
    answers, prompts = scripted
    answers.extend(["2", "sk-secret-value"])
    await commands.dispatch(app, "/user")

    key_prompt = next(p for p in prompts if "API key" in p)
    assert key_prompt.startswith("[password] ")
    assert "sk-secret-value" not in text_of(app.console)


async def test_a_stored_key_is_shown_masked(app, scripted, fake_keyring):
    cli_settings.set_api_key("sk-abcdefghijkl-TAIL")
    await commands.dispatch(app, "/user")
    out = text_of(app.console)

    assert "TAIL" in out, "the last four characters identify which key is set"
    assert "sk-abcdefghijkl" not in out


async def test_an_empty_answer_changes_nothing(app, scripted, fake_keyring):
    cli_settings.set_user_name("Original")
    answers, _ = scripted
    answers.extend(["1", ""])
    await commands.dispatch(app, "/user")
    assert cli_settings.get_user_name() == "Original"


async def test_user_rejects_an_option_outside_one_and_two(app, scripted, fake_keyring):
    answers, _ = scripted
    answers.append("3")
    await commands.dispatch(app, "/user")
    assert "Not a valid choice" in text_of(app.console)


# ── Command dispatch ─────────────────────────────────────────────────────────


async def test_help_lists_every_registered_command(app):
    await commands.dispatch(app, "/help")
    out = text_of(app.console)
    for command in commands.COMMANDS:
        assert command.name in out


async def test_exit_ends_the_loop(app):
    assert await commands.dispatch(app, "/exit") is False


@pytest.mark.parametrize("alias", ["/quit", "/q"])
async def test_exit_aliases_work(app, alias):
    assert await commands.dispatch(app, alias) is False


async def test_an_unknown_command_is_reported_not_sent_to_the_agent(app):
    """A typo should cost nothing — it must not reach a paid model."""
    assert await commands.dispatch(app, "/mdoel") is True
    assert "Unknown command" in text_of(app.console)


async def test_skip_without_a_pending_question_says_so(app):
    await commands.dispatch(app, "/skip")
    assert "Nothing to skip" in text_of(app.console)


# ── Rendering ────────────────────────────────────────────────────────────────


def test_a_skipped_check_does_not_read_as_a_pass(console):
    """
    The whole point of CheckStatus.SKIPPED: a missing linter must never be
    reported as a clean result.
    """
    renderer = EventRenderer(console)
    renderer.handle({
        "type": "validation_result", "phase": "lint",
        "passed": True, "status": "skipped", "output": "ruff is not installed.",
    })
    out = text_of(console)

    assert "skipped" in out
    assert "passed" not in out
    assert "ruff is not installed." in out, "a skipped check owes the user a reason"


def test_a_passing_check_is_not_padded_with_output(console):
    renderer = EventRenderer(console)
    renderer.handle({
        "type": "validation_result", "phase": "test",
        "passed": True, "status": "passed", "output": "4 passed",
    })
    out = text_of(console)
    assert "passed" in out
    assert "4 passed" not in out


def test_a_failing_check_shows_its_errors(console):
    renderer = EventRenderer(console)
    renderer.handle({
        "type": "validation_result", "phase": "lint",
        "passed": False, "status": "failed", "output": "E501 line too long",
    })
    out = text_of(console)
    assert "failed" in out
    assert "E501 line too long" in out


def test_markup_in_validation_output_is_not_interpreted(console):
    """Tool output is data. A bracket in a traceback is a bracket."""
    renderer = EventRenderer(console)
    renderer.handle({
        "type": "validation_result", "phase": "lint", "status": "failed",
        "output": "expected list[str], got [bold]",
    })
    out = text_of(console)
    assert "expected list[str], got [bold]" in out


def test_an_unrecognised_status_is_not_treated_as_success(console):
    renderer = EventRenderer(console)
    renderer.handle({
        "type": "validation_result", "phase": "lint", "status": "weird", "output": "",
    })
    assert "passed" not in text_of(console)


def test_plan_tokens_are_streamed_in_order(console):
    renderer = EventRenderer(console)
    for chunk in ["1. read ", "main.py", "\n2. patch it"]:
        renderer.handle({"type": "plan_chunk", "chunk": chunk})
    renderer.finish()

    out = text_of(console)
    assert "1. read main.py" in out
    assert "2. patch it" in out


def test_markup_in_model_output_is_not_interpreted(console):
    """A stray bracket in streamed output must not be parsed as a style tag."""
    renderer = EventRenderer(console)
    renderer.handle({"type": "plan_chunk", "chunk": "use list[str] not [bold]"})
    renderer.finish()
    assert "list[str] not [bold]" in text_of(console)


def test_a_tool_call_shows_the_tool_and_its_target(console):
    renderer = EventRenderer(console)
    renderer.handle({"type": "tool_call", "tool": "patch_file", "args": {"path": "core/x.py"}})
    out = text_of(console)
    assert "patch_file" in out
    assert "core/x.py" in out


def test_a_question_is_labelled_with_its_turn(console):
    renderer = EventRenderer(console)
    renderer.handle({"type": "question", "question": "Which file?", "turn": 2})
    out = text_of(console)
    assert "Clarifying question 2" in out
    assert "Which file?" in out
    assert "/skip" in out, "the user needs to know they can bail out"


def test_a_refined_prompt_is_shown_only_after_interrogation(console):
    renderer = EventRenderer(console)
    renderer.handle({"type": "ready", "refined_prompt": "echo", "did_interrogate": False})
    assert text_of(console) == ""

    renderer.handle({"type": "ready", "refined_prompt": "a full spec", "did_interrogate": True})
    assert "a full spec" in text_of(console)


def test_the_summary_is_not_printed_twice(console):
    """`done.summary` repeats `impl_done.content`; rendering both doubles it."""
    renderer = EventRenderer(console)
    renderer.handle({"type": "impl_done", "content": "Added retries to the uploader."})
    renderer.handle({"type": "done", "summary": "Added retries to the uploader."})

    assert text_of(console).count("Added retries to the uploader.") == 1


def test_an_error_is_rendered_prominently(console):
    renderer = EventRenderer(console)
    renderer.handle({"type": "error", "message": "No OpenAI API key configured"})
    assert "No OpenAI API key configured" in text_of(console)


def test_an_unknown_event_type_is_ignored(console):
    """Forward compatibility: a new backend event must not crash the frontend."""
    renderer = EventRenderer(console)
    renderer.handle({"type": "something_new", "payload": 1})
    assert text_of(console) == ""


# ── Conversation statelessness ───────────────────────────────────────────────


@pytest.fixture
def captured_sessions(monkeypatch):
    """Record every ChatSession the app builds, and stub out the agent work."""
    built: list[dict] = []

    class StubSession:
        def __init__(self, project_path, project_index=None, model=None, history=None):
            built.append(
                {"project_path": project_path, "model": model, "history": list(history or [])}
            )
            self.finished = False
            self.awaiting_answer = False
            self.answers: list[str] = []

        # Mirrors the real coder: the reply arrives as `impl_done`, and `done`
        # repeats it as a summary.
        async def start(self, message):
            self.finished = True
            yield {"type": "impl_done", "content": "stubbed reply"}
            yield {"type": "done", "summary": "stubbed reply"}

        async def answer(self, message):
            self.answers.append(message)
            self.finished = True
            yield {"type": "impl_done", "content": "stubbed reply"}
            yield {"type": "done", "summary": "stubbed reply"}

        async def force_ready(self):
            self.finished = True
            yield {"type": "impl_done", "content": "forced"}
            yield {"type": "done", "summary": "forced"}

    monkeypatch.setattr("cli.app.ChatSession", StubSession)
    return built


async def test_every_request_starts_a_brand_new_session(app, captured_sessions):
    """The session object is single-use; the conversation outlives it."""
    await app._send("first")
    await app._send("second")

    assert len(captured_sessions) == 2
    assert app._session is None, "a finished session must be dropped, not reused"


async def test_an_answer_continues_the_open_session(app, captured_sessions):
    """A reply to a clarifying question is not a new request."""
    await app._send("vague request")
    assert len(captured_sessions) == 1

    # Re-open the session as if a clarifying question were pending.
    class Pending:
        finished = False
        awaiting_answer = True

        def __init__(self):
            self.answered: list[str] = []

        async def answer(self, message):
            self.answered.append(message)
            self.finished = True
            yield {"type": "done", "summary": message}

    session = Pending()
    app._session = session
    await app._send("api/auth.py")

    assert session.answered == ["api/auth.py"]
    assert len(captured_sessions) == 1, "no second session should have been built"


async def test_a_follow_up_carries_the_earlier_turns(app, captured_sessions):
    """
    The reported bug: "what is this project about" → a good answer → "sure" →
    "I'm not sure what you want me to do yet." The second request must be able
    to see the first.
    """
    await app._send("what is this project about")
    await app._send("sure")

    carried = captured_sessions[1]["history"]
    assert [t["content"] for t in carried] == [
        "what is this project about",
        "stubbed reply",
    ]
    assert [t["role"] for t in carried] == ["user", "assistant"]


async def test_the_current_request_is_not_duplicated_into_its_own_history(
    app, captured_sessions
):
    assert_empty = await app._send("first thing")
    assert captured_sessions[0]["history"] == [], assert_empty


async def test_a_clarifying_question_is_remembered_as_a_turn(app, captured_sessions):
    """An interrogation is part of the conversation, not scaffolding."""
    app._record({"type": "question", "question": "Which file?", "turn": 1})
    assert app._history == [{"role": "assistant", "content": "Which file?"}]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "plan", "content": "1. do a thing"},
        {"type": "plan_chunk", "chunk": "1. do"},
        {"type": "tool_call", "tool": "read_file", "args": {"path": "a.py"}},
        {"type": "validation_result", "phase": "lint", "status": "passed", "output": ""},
        {"type": "error", "message": "boom"},
        {"type": "impl_done", "content": "   "},
    ],
)
async def test_working_notes_are_not_recorded_as_conversation(app, event):
    """Plans, tool calls and validation output are scaffolding, not dialogue."""
    app._record(event)
    assert app._history == []


async def test_clearing_forgets_the_conversation(app, captured_sessions, fake_keyring):
    await app._send("first")
    assert app._history

    await commands.dispatch(app, "/clear")
    assert app._history == []

    await app._send("second")
    assert captured_sessions[-1]["history"] == [], "nothing may survive a /clear"


async def test_the_selected_model_is_used_for_the_next_request(app, captured_sessions):
    await commands.dispatch(app, "/model gpt-5.4-mini")
    await app._send("do it")
    assert captured_sessions[-1]["model"] == "gpt-5.4-mini"


async def test_awaiting_answer_reflects_the_open_session(app):
    assert app.awaiting_answer is False


# ── Indexing ─────────────────────────────────────────────────────────────────


async def test_an_already_embedded_project_is_not_re_embedded(console, monkeypatch, tmp_path):
    """Re-embedding an unchanged project costs money and buys nothing."""
    monkeypatch.setattr(indexing, "collection_size", lambda path: 42)

    async def explode(path):
        raise AssertionError("embedding must be skipped")
        yield  # pragma: no cover

    monkeypatch.setattr(indexing, "embed_project", explode)
    assert await indexing.embed(console, str(tmp_path)) is True
    assert "42 chunks already embedded" in text_of(console)


async def test_forcing_re_embeds_even_when_vectors_exist(console, monkeypatch, tmp_path):
    monkeypatch.setattr(indexing, "collection_size", lambda path: 42)
    calls: list[str] = []

    async def fake_embed(path):
        calls.append(path)
        yield {"step": "A", "status": "done", "files": 3}
        yield {"step": "done"}

    monkeypatch.setattr(indexing, "embed_project", fake_embed)
    await indexing.embed(console, str(tmp_path), force=True)
    assert calls == [str(tmp_path)]


async def test_an_indexing_error_points_at_the_fix(console, monkeypatch, tmp_path):
    monkeypatch.setattr(indexing, "collection_size", lambda path: 0)

    async def failing(path):
        yield {"step": "error", "message": "No OpenAI API key configured"}

    monkeypatch.setattr(indexing, "embed_project", failing)
    assert await indexing.embed(console, str(tmp_path)) is False

    out = text_of(console)
    assert "No OpenAI API key configured" in out
    assert "/user" in out, "the error should name the command that fixes it"


def test_a_missing_path_is_reported_not_raised(console, tmp_path):
    assert indexing.scan(console, tmp_path / "nope") is None
    assert "Path not found" in text_of(console)


def test_a_file_path_is_rejected(console, tmp_path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    assert indexing.scan(console, target) is None
    assert "not a directory" in text_of(console).lower()


def test_the_header_reports_files_languages_and_git(console, tmp_project):
    index = indexing.scan(console, tmp_project)
    assert index is not None
    indexing.describe(console, index)

    out = text_of(console)
    assert str(tmp_project) in out
    assert "files" in out
    assert "Python" in out


# ── Entry point ──────────────────────────────────────────────────────────────


def test_the_path_defaults_to_the_current_directory():
    assert cli_main._parse_args([]).path == "."


def test_a_path_argument_is_taken_verbatim():
    assert cli_main._parse_args(["~/code/app"]).path == "~/code/app"


def test_the_model_defaults_to_the_shared_default():
    assert cli_main._parse_args([]).model == _DEFAULT_MODEL


def test_every_available_model_is_accepted_on_the_command_line():
    for model in AVAILABLE_MODELS:
        assert cli_main._parse_args(["--model", model]).model == model


def test_an_unknown_model_is_refused_at_the_command_line(capsys):
    """argparse should reject it before any work — and name the valid ids."""
    with pytest.raises(SystemExit) as exc:
        cli_main._parse_args(["--model", "gpt-9-ultra"])
    assert exc.value.code == 2
    assert "gpt-5.4-mini" in capsys.readouterr().err


def test_reindex_and_verbose_default_to_off():
    args = cli_main._parse_args([])
    assert args.reindex is False
    assert args.verbose is False


def test_reindex_and_verbose_can_be_set():
    args = cli_main._parse_args(["--reindex", "--verbose"])
    assert args.reindex is True
    assert args.verbose is True


def test_the_exit_code_comes_from_the_app(monkeypatch, tmp_path):
    async def fake_run(self, *, reindex=False):
        return 7

    monkeypatch.setattr(InterroCLI, "run", fake_run)
    assert cli_main.run([str(tmp_path)]) == 7


def test_ctrl_c_exits_without_a_traceback(monkeypatch, tmp_path):
    async def boom(self, *, reindex=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(InterroCLI, "run", boom)
    assert cli_main.run([str(tmp_path)]) == 130


def test_reindex_reaches_the_app(monkeypatch, tmp_path):
    seen: dict = {}

    async def fake_run(self, *, reindex=False):
        seen["reindex"] = reindex
        return 0

    monkeypatch.setattr(InterroCLI, "run", fake_run)
    cli_main.run([str(tmp_path), "--reindex"])
    assert seen["reindex"] is True


# ── Terminal chrome ──────────────────────────────────────────────────────────


def test_the_banner_greets_a_known_user(app, fake_keyring):
    cli_settings.set_user_name("Alex")
    app.print_banner()
    assert "Welcome back, Alex" in text_of(app.console)


def test_the_banner_falls_back_when_no_name_is_set(app, fake_keyring):
    app.print_banner()
    assert "Welcome to InterroAI" in text_of(app.console)


def test_the_toolbar_shows_the_project_and_model(app):
    toolbar = app._toolbar().value
    assert Path(app.project_path).name in toolbar
    assert model_label(app.model) in toolbar


def test_the_toolbar_switches_while_a_question_is_pending(app):
    class Pending:
        awaiting_answer = True
        finished = False

    app._session = Pending()
    assert "/skip" in app._toolbar().value


async def test_clear_redraws_the_banner(app, fake_keyring):
    await commands.dispatch(app, "/clear")
    assert "Welcome to InterroAI" in text_of(app.console)


# ── Project payload ──────────────────────────────────────────────────────────


def test_an_unindexed_project_passes_an_empty_context(app):
    assert app._index_payload() == {}


def test_the_index_is_passed_to_the_agent_as_a_plain_dict(app, tmp_project):
    app.project_index = indexing.scan(app.console, tmp_project)
    payload = app._index_payload()

    assert isinstance(payload, dict)
    assert payload["file_tree"]["kind"] == "dir"
    assert "git_context" in payload


# ── First-run setup ──────────────────────────────────────────────────────────


async def test_setup_is_skipped_when_a_key_already_exists(app, scripted, fake_keyring):
    cli_settings.set_api_key("sk-existing")
    _, prompts = scripted
    await app._first_run_setup()
    assert prompts == [], "an existing key must not be asked for again"


async def test_setup_stores_a_key_given_on_first_run(app, scripted, fake_keyring):
    answers, prompts = scripted
    answers.append("sk-brand-new")
    await app._first_run_setup()

    assert fake_keyring[("interroai", "openai_api_key")] == "sk-brand-new"
    assert prompts[0].startswith("[password] "), "the key must be read masked"


async def test_setup_can_be_skipped(app, scripted, fake_keyring):
    answers, _ = scripted
    answers.append("")
    await app._first_run_setup()

    assert not cli_settings.has_api_key()
    assert "/user" in text_of(app.console), "the user needs to know how to set it later"


# ── /skip plumbing ───────────────────────────────────────────────────────────


async def test_force_ready_without_a_session_does_nothing(app):
    await app.force_ready()  # must not raise


async def test_skip_forwards_to_the_session(app):
    class Pending:
        awaiting_answer = True
        finished = False

        def __init__(self):
            self.forced = False

        async def force_ready(self):
            self.forced = True
            self.finished = True
            yield {"type": "done", "summary": "forced"}

    session = Pending()
    app._session = session
    await commands.dispatch(app, "/skip")

    assert session.forced is True
    assert app._session is None, "a finished session must be dropped"
