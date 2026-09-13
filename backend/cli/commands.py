"""
Slash commands.

Every command is registered here with the help text `/help` prints, so the
listing can never drift from what actually runs. A handler returns True to
keep the session going and False to end it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.session import AVAILABLE_MODELS
from cli import indexing, settings
from cli.theme import BULLET, PASS, PROMPT, model_label

if TYPE_CHECKING:
    from cli.app import InterroCLI


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    handler: Callable[[InterroCLI, str], Awaitable[bool]]


# ── /help ────────────────────────────────────────────────────────────────────


async def cmd_help(app, _args: str) -> bool:
    app.console.print(f"\n[label]{BULLET} Commands[/label]")
    width = max(len(c.name) for c in COMMANDS)
    for command in COMMANDS:
        app.console.print(f"  [tool]{command.name:<{width}}[/tool]  [hint]{command.summary}[/hint]")
    app.console.print(
        "\n  [hint]Anything else is sent to the agent. "
        "Ctrl-C interrupts, Ctrl-D exits.[/hint]"
    )
    return True


# ── /model ───────────────────────────────────────────────────────────────────


async def cmd_model(app, args: str) -> bool:
    """Pick the model by number, or name it directly: `/model gpt-5.4-mini`."""
    choice = args.strip()

    if choice:
        if choice not in AVAILABLE_MODELS:
            app.console.print(f"[fail]Unknown model {choice!r}.[/fail]")
            return True
        app.model = choice
        app.console.print(f"[ok]{PASS} Model set to {model_label(choice)}[/ok]")
        return True

    app.console.print(f"\n[label]{BULLET} Select a model[/label]")
    for i, model in enumerate(AVAILABLE_MODELS, start=1):
        marker = "  [hint]← current[/hint]" if model == app.model else ""
        app.console.print(f"  [tool]{i}.[/tool] {model_label(model)}{marker}")

    answer = await app.ask(f"\n  Choose 1-{len(AVAILABLE_MODELS)} (Enter to keep current){PROMPT}")
    if not answer.strip():
        return True

    selected = _pick(answer, AVAILABLE_MODELS)
    if selected is None:
        app.console.print("[fail]Not a valid choice.[/fail]")
        return True

    app.model = selected
    app.console.print(f"[ok]{PASS} Model set to {model_label(selected)}[/ok]")
    return True


# ── /user ────────────────────────────────────────────────────────────────────


async def cmd_user(app, _args: str) -> bool:
    """The two-item settings menu: username and OpenAI API key."""
    name = settings.get_user_name() or "[hint]not set[/hint]"
    key = settings.masked_api_key() or "[hint]not set[/hint]"

    app.console.print(f"\n[label]{BULLET} Settings[/label]")
    app.console.print(f"  [tool]1.[/tool] Username         {name}")
    app.console.print(f"  [tool]2.[/tool] OpenAI API key   {key}")

    answer = (await app.ask(f"\n  Choose 1-2 (Enter to cancel){PROMPT}")).strip()
    if not answer:
        return True

    if answer == "1":
        value = (await app.ask(f"  Username{PROMPT}")).strip()
        if not value:
            app.console.print("[hint]Unchanged.[/hint]")
            return True
        settings.set_user_name(value)
        app.console.print(f"[ok]{PASS} Username saved[/ok]")

    elif answer == "2":
        # Masked input: the key must not land in the terminal scrollback.
        value = (await app.ask(f"  OpenAI API key{PROMPT}", is_password=True)).strip()
        if not value:
            app.console.print("[hint]Unchanged.[/hint]")
            return True
        settings.set_api_key(value)
        app.console.print(f"[ok]{PASS} API key saved to the OS keychain[/ok]")

    else:
        app.console.print("[fail]Not a valid choice.[/fail]")

    return True


# ── /index ───────────────────────────────────────────────────────────────────


async def cmd_index(app, _args: str) -> bool:
    app.console.print()
    index = indexing.scan(app.console, app.project_path)
    if index is None:
        return True
    app.project_index = index
    await indexing.embed(app.console, index.path, force=True)
    return True


# ── /clear ───────────────────────────────────────────────────────────────────


async def cmd_clear(app, _args: str) -> bool:
    """Clear the screen and start the conversation over."""
    app.clear_conversation()
    app.console.clear()
    app.print_banner()
    app.console.print("  [hint]Conversation cleared.[/hint]\n")
    return True


# ── /exit ────────────────────────────────────────────────────────────────────


async def cmd_exit(app, _args: str) -> bool:
    return False


COMMANDS: tuple[Command, ...] = (
    Command("/help", "Show this list", cmd_help),
    Command("/model", "Choose the model for the next request", cmd_model),
    Command("/user", "Set your username or OpenAI API key", cmd_user),
    Command("/index", "Re-index the project for semantic search", cmd_index),
    Command("/clear", "Clear the screen and forget the conversation", cmd_clear),
    Command("/exit", "Quit", cmd_exit),
)

_ALIASES = {"/quit": "/exit", "/q": "/exit", "/?": "/help"}

_BY_NAME = {c.name: c for c in COMMANDS}

#: Names offered by tab completion, aliases included.
COMMAND_NAMES: tuple[str, ...] = tuple([c.name for c in COMMANDS] + list(_ALIASES))


def _pick(answer: str, options) -> str | None:
    """Resolve a 1-based menu answer, tolerating the option's own name."""
    answer = answer.strip()
    if answer in options:
        return answer
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return None


def lookup(name: str) -> Command | None:
    return _BY_NAME.get(_ALIASES.get(name, name))


async def dispatch(app, line: str) -> bool:
    """
    Run the slash command in *line*. Returns False when the CLI should exit.

    An unknown command is reported rather than forwarded to the agent: a typo'd
    `/mdoel` is a mistake to correct, not a prompt worth paying a model to
    interpret.
    """
    name, _, args = line.strip().partition(" ")
    command = lookup(name)
    if command is None:
        app.console.print(f"[fail]Unknown command {name}[/fail] [hint]— try /help[/hint]")
        return True
    return await command.handler(app, args)
