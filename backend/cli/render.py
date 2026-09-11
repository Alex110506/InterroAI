"""
Agent events → terminal output.

The renderer is the CLI's whole view layer. It consumes exactly the event
dicts `agents/session.py` emits, so anything the WebSocket client could show,
the terminal can show too.

Rendering choices worth knowing:

  * Plan tokens are streamed as they arrive, unstyled. A live-updating
    Markdown block looks better for three seconds and then starts fighting the
    scrollback once the plan is longer than the window; raw streamed text
    always behaves.
  * The implementation summary *is* rendered as Markdown — it arrives whole,
    so there is nothing to fight.
  * A validation result reports `status`, not `passed`. "skipped" and "passed"
    are different outcomes and the whole point of `CheckStatus` is that a
    missing linter never reads as a clean bill of health.
"""
from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

from cli.theme import BULLET, FAIL, PASS, SKIP

#: Validation states → (glyph, style). Anything unrecognised is treated as a
#: failure: an unknown outcome is not evidence that a check succeeded.
_VALIDATION_STYLES = {
    "passed": (PASS, "ok"),
    "failed": (FAIL, "fail"),
    "skipped": (SKIP, "skip"),
}

_MAX_VALIDATION_OUTPUT = 800


class EventRenderer:
    """Stateful because streamed plan tokens need to know when a line is open."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._streaming_plan = False

    # ── Entry point ───────────────────────────────────────────────────────

    def handle(self, event: dict) -> None:
        handler = getattr(self, f"_on_{event.get('type', '')}", None)
        if handler is not None:
            handler(event)

    def finish(self) -> None:
        """Close any half-written streamed line before the prompt returns."""
        self._end_plan_stream()

    # ── Individual events ─────────────────────────────────────────────────

    def _on_ready(self, event: dict) -> None:
        # Only worth showing when interrogation actually changed the request;
        # otherwise it just echoes what the user typed a second ago.
        if not event.get("did_interrogate"):
            return
        self._end_plan_stream()
        self._console.print(f"\n[label]{BULLET} Refined request[/label]")
        self._console.print(f"  [hint]{event.get('refined_prompt', '')}[/hint]")
        self._console.print()

    def _on_question(self, event: dict) -> None:
        self._end_plan_stream()
        turn = event.get("turn", 1)
        self._console.print(f"\n[label]{BULLET} Clarifying question {turn}[/label]")
        self._console.print(f"  {event.get('question', '')}")
        self._console.print("  [hint]/skip to stop asking and implement[/hint]")

    def _on_plan_chunk(self, event: dict) -> None:
        chunk = event.get("chunk") or ""
        if not chunk:
            return
        if not self._streaming_plan:
            self._console.print(f"\n[label]{BULLET} Plan[/label]")
            self._streaming_plan = True
        # `end=""` and no markup: this is model output mid-token, and a stray
        # bracket in it must not be parsed as a style tag.
        self._console.print(chunk, end="", markup=False, highlight=False)

    def _on_tool_call(self, event: dict) -> None:
        self._end_plan_stream()
        args = event.get("args") or {}
        detail = args.get("path") or args.get("pattern") or args.get("query") or ""
        line = f"[tool]{BULLET} {event.get('tool', '?')}[/tool]"
        if detail:
            line += f"  [arg]{detail}[/arg]"
        self._console.print(line)

    def _on_impl_done(self, event: dict) -> None:
        self._end_plan_stream()
        content = (event.get("content") or "").strip()
        if not content:
            return
        self._console.print()
        self._console.print(Markdown(content))

    def _on_validation_result(self, event: dict) -> None:
        self._end_plan_stream()
        phase = "Linter" if event.get("phase") == "lint" else "Tests"
        status = event.get("status") or ("passed" if event.get("passed") else "failed")
        glyph, style = _VALIDATION_STYLES.get(status, (FAIL, "fail"))

        self._console.print(f"[{style}]{glyph} {phase} — {status}[/{style}]")

        # A passing check has nothing to report; a skipped one owes the user a
        # reason ("ruff is not installed"), and a failing one owes the errors.
        output = (event.get("output") or "").strip()
        if status != "passed" and output:
            if len(output) > _MAX_VALIDATION_OUTPUT:
                output = output[:_MAX_VALIDATION_OUTPUT] + "\n…"
            for line in output.splitlines():
                # `markup=False` because tool output is not ours to parse — a
                # bracket in a traceback is a bracket, not a style tag. The
                # style therefore has to be passed, not embedded.
                self._console.print(
                    f"  {line}", style="hint", markup=False, highlight=False
                )

    def _on_correction(self, event: dict) -> None:
        self._end_plan_stream()
        self._console.print(
            f"[warn]{BULLET} Self-correcting — attempt {event.get('attempt', '?')}[/warn]"
        )

    def _on_done(self, event: dict) -> None:
        self._end_plan_stream()
        # The summary has already been printed as `impl_done`; repeating it
        # here would double every answer.
        self._console.print(f"\n[ok]{PASS} Done[/ok]")

    def _on_error(self, event: dict) -> None:
        self._end_plan_stream()
        self._console.print(f"\n[fail]{FAIL} {event.get('message', 'Unknown error')}[/fail]")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _end_plan_stream(self) -> None:
        """Close the open streamed line, then separate it from what follows."""
        if self._streaming_plan:
            self._console.print()
            self._console.print()
            self._streaming_plan = False
