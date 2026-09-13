"""
`interroai` — command-line entry point.

    interroai                 index and open the current directory
    interroai ~/code/myapp    index and open that path
    interroai --reindex       force a fresh embedding pass
    interroai --model gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console

from agents.session import _DEFAULT_MODEL, AVAILABLE_MODELS
from cli.app import InterroCLI
from cli.theme import THEME


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="interroai",
        description="AI coding agent for a local project.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project directory to index and work in (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        choices=list(AVAILABLE_MODELS),
        help=f"Model to start with (default: {_DEFAULT_MODEL}). Change it later with /model.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-embed the project even if it already has stored vectors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log agent activity to stderr.",
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    """
    Quiet by default.

    The agent logs every tool call and event at INFO; interleaved with the
    rendered transcript that is unreadable, so the transcript is the default
    view and `--verbose` is the way to see the machinery.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    console = Console(theme=THEME)
    app = InterroCLI(
        project_path=str(Path(args.path).expanduser()),
        console=console,
        model=args.model,
    )

    try:
        return asyncio.run(app.run(reindex=args.reindex))
    except KeyboardInterrupt:
        # Ctrl-C outside the prompt's own handling — leave without a traceback.
        console.print("\n[hint]Bye.[/hint]")
        return 130


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
