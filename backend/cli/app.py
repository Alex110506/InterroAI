"""
The interactive loop.

One project, one terminal. A run indexes the path it was given, then reads
requests until the user leaves.

The conversation lives here, in memory, for exactly as long as the process
does. Each request builds a fresh `ChatSession` and hands it the turns so far,
so a follow-up like "sure" or "now the other file" resolves against what was
already said. Nothing is loaded at startup and nothing is written at exit:
opening a project always gives you an empty conversation, and `/clear` empties
it again mid-run.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from agents.session import _DEFAULT_MODEL, ChatSession
from cli import commands, indexing, settings
from cli.render import EventRenderer
from cli.theme import BANNER, BULLET, PASS, PROMPT, model_label


class InterroCLI:
    """Owns the terminal, the current project and the current model."""

    def __init__(
        self,
        project_path: str,
        console: Console,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.console = console
        self.project_path = project_path
        self.project_index = None
        self.model = model

        self._session: ChatSession | None = None
        #: Turns so far, in OpenAI message shape. In memory only — never read
        #: from or written to disk.
        self._history: list[dict] = []
        self._renderer = EventRenderer(console)
        self._prompt = PromptSession(
            history=InMemoryHistory(),
            completer=WordCompleter(list(commands.COMMAND_NAMES), sentence=True),
            complete_while_typing=True,
        )

    # ── State the commands read ───────────────────────────────────────────

    @property
    def awaiting_answer(self) -> bool:
        """True while the grill agent is waiting on an answer from the user."""
        return self._session is not None and self._session.awaiting_answer

    # ── Terminal chrome ───────────────────────────────────────────────────

    def print_banner(self) -> None:
        name = settings.get_user_name()
        greeting = f"Welcome back, {name}" if name else "Welcome to InterroAI"
        self.console.print(f"\n[brand]{BANNER} {greeting}[/brand]")
        self.console.print("  [hint]/help for commands · /model to switch model[/hint]\n")

    def _toolbar(self) -> HTML:
        project = Path(self.project_path).name
        if self.awaiting_answer:
            return HTML(f"  <b>{project}</b>  ·  answering question  ·  /skip to implement now")
        return HTML(f"  <b>{project}</b>  ·  {model_label(self.model)}  ·  /help")

    # ── Input ─────────────────────────────────────────────────────────────

    async def ask(self, message: str, *, is_password: bool = False) -> str:
        """
        Read one line, for a command's own sub-prompt.

        Rich owns the styled output and prompt_toolkit owns the input, so the
        message is printed first and prompt_toolkit is handed a bare caret.
        """
        self.console.print(message, end="")
        with patch_stdout():
            return await self._prompt.prompt_async("", is_password=is_password)

    async def _read_request(self) -> str:
        with patch_stdout():
            return await self._prompt.prompt_async(
                PROMPT,
                bottom_toolbar=self._toolbar,
                # Explicit: a PromptSession remembers the arguments of the last
                # prompt(), so the masked API-key prompt would otherwise leave
                # every later request echoing as asterisks.
                is_password=False,
            )

    # ── Agent plumbing ────────────────────────────────────────────────────

    async def _consume(self, stream: AsyncIterator[dict]) -> None:
        """
        Render an event stream, with a spinner filling the gaps.

        The spinner is stopped before each event and restarted after, so
        printed output never lands on top of a live-updating line.

        Anything the agent *said* — a clarifying question, a final summary —
        is recorded as a turn on the way past, so the next request can refer
        back to it. Plans, tool calls and validation results are working notes,
        not conversation, and are not recorded.
        """
        status = self.console.status("[hint]Thinking…[/hint]", spinner="dots")
        status.start()
        try:
            async for event in stream:
                status.stop()
                self._renderer.handle(event)
                self._record(event)
                status.start()
        except asyncio.CancelledError:
            raise
        finally:
            status.stop()
            self._renderer.finish()

    def _record(self, event: dict) -> None:
        spoken = {"question": "question", "impl_done": "content"}.get(event.get("type"))
        if spoken is None:
            return
        content = (event.get(spoken) or "").strip()
        if content:
            self._history.append({"role": "assistant", "content": content})

    async def _send(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

        if self._session is not None and self._session.awaiting_answer:
            stream = self._session.answer(text)
        else:
            # A new request gets a new session, seeded with the conversation so
            # far. The session itself stays single-use; the transcript is ours.
            self._session = ChatSession(
                project_path=self.project_path,
                project_index=self._index_payload(),
                model=self.model,
                history=self._history[:-1],
            )
            stream = self._session.start(text)

        await self._consume(stream)

        if self._session is not None and self._session.finished:
            self._session = None

    async def force_ready(self) -> None:
        """Backs `/skip`."""
        if self._session is None:
            return
        await self._consume(self._session.force_ready())
        if self._session.finished:
            self._session = None

    def clear_conversation(self) -> None:
        """Drop the transcript. Backs `/clear`."""
        self._history.clear()
        self._session = None

    def _index_payload(self) -> dict:
        """The ProjectIndex as the agents expect it — a plain dict, or empty."""
        return self.project_index.model_dump() if self.project_index is not None else {}

    # ── Startup ───────────────────────────────────────────────────────────

    async def _first_run_setup(self) -> None:
        """
        Without a key nothing works, so ask for one before indexing rather
        than letting every subsequent step fail in its own way.
        """
        if settings.has_api_key():
            return
        self.console.print(f"[warn]{BULLET} No OpenAI API key is configured.[/warn]")
        key = await self.ask(f"  OpenAI API key (Enter to skip){PROMPT}", is_password=True)
        key = key.strip()
        if key:
            settings.set_api_key(key)
            self.console.print(f"[ok]{PASS} Saved to the OS keychain[/ok]\n")
        else:
            self.console.print("  [hint]Set one later with /user.[/hint]\n")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self, *, reindex: bool = False) -> int:
        self.print_banner()
        await self._first_run_setup()

        self.project_index = await indexing.run(
            self.console, self.project_path, force=reindex
        )
        if self.project_index is None:
            return 1

        # Resolved form, so the agent's sandbox and the vector store agree.
        self.project_path = self.project_index.path
        self.console.print()

        while True:
            try:
                text = (await self._read_request()).strip()
            except KeyboardInterrupt:
                continue          # Ctrl-C on an empty prompt: clear the line
            except EOFError:
                break             # Ctrl-D: leave

            if not text:
                continue

            if text.startswith("/"):
                if not await commands.dispatch(self, text):
                    break
                continue

            try:
                await self._send(text)
            except KeyboardInterrupt:
                # Mid-request Ctrl-C. The session is mid-flight and cannot be
                # resumed, so drop it and return to a clean prompt.
                self._session = None
                self.console.print("\n[hint]Interrupted.[/hint]")

        self.console.print("\n[hint]Bye.[/hint]")
        return 0
