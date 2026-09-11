"""
Startup indexing for the CLI.

Phase 1 (file tree + git context) is cheap and always runs. Phase 2 (chunk →
embed → store) costs real money per run, and unlike the desktop app — where a
project was opened once and stayed open — a CLI is relaunched constantly. So
Phase 2 is skipped when the project already has vectors stored, unless the
caller forces it with `--reindex` or `/index`.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from cli.theme import FAIL, PASS, SKIP
from core.project_index import ProjectIndex, ProjectPathError, build_index, embed_project
from core.vector_store import collection_size

_STEP_LABELS = {
    "A": "Scanned",
    "B": "Chunked",
    "C": "Embedded",
    "D": "Stored",
}


def scan(console: Console, path: str | Path) -> ProjectIndex | None:
    """Phase 1. Returns None (having explained why) if the path is unusable."""
    try:
        with console.status("[hint]Scanning project…[/hint]", spinner="dots"):
            return build_index(path)
    except ProjectPathError as exc:
        console.print(f"[fail]{FAIL} {exc}[/fail]")
        return None


def describe(console: Console, index: ProjectIndex) -> None:
    """The header block: where we are, what's here, what git thinks."""
    git = index.git_context
    console.print(f"  [hint]path[/hint]   {index.path}")

    facts = [f"{index.total_files} files"]
    if index.languages:
        facts.append(" · ".join(index.languages[:4]))
    console.print(f"  [hint]code[/hint]   {' · '.join(facts)}")

    if git.is_git_repo:
        state = [git.branch or "detached"]
        if git.modified_files:
            state.append(f"{len(git.modified_files)} modified")
        if git.untracked_files:
            state.append(f"{len(git.untracked_files)} untracked")
        console.print(f"  [hint]git[/hint]    {' · '.join(state)}")


async def embed(console: Console, path: str, *, force: bool = False) -> bool:
    """
    Phase 2. Returns True if the project has usable vectors afterwards.

    A failure here is reported but not fatal: the grill agent degrades to
    tree + git context without RAG, and the user may simply need to set an API
    key with `/user` and re-run `/index`.
    """
    existing = collection_size(path)
    if existing and not force:
        console.print(
            f"  [hint]index[/hint]  {existing} chunks already embedded "
            f"[hint](/index to refresh)[/hint]"
        )
        return True

    failed = False
    with console.status("[hint]Indexing…[/hint]", spinner="dots") as status:
        async for event in embed_project(path):
            step = event.get("step")

            if step == "error":
                status.stop()
                console.print(f"  [fail]{FAIL} Indexing failed — {event.get('message')}[/fail]")
                console.print("  [hint]Set an API key with /user, then run /index[/hint]")
                failed = True
                break

            if step == "done":
                break

            if event.get("status") == "progress" and step == "C":
                status.update(
                    f"[hint]Embedding {event.get('embedded')}/{event.get('total')}…[/hint]"
                )
            elif event.get("status") == "done":
                count = (
                    event.get("files")
                    or event.get("chunks")
                    or event.get("stored")
                    or event.get("total")
                    or 0
                )
                status.stop()
                label = _STEP_LABELS.get(step, step)
                console.print(f"  [ok]{PASS}[/ok] [hint]{label} {count}[/hint]")
                status.start()

    if failed:
        return False

    if not collection_size(path):
        console.print(f"  [skip]{SKIP} Nothing indexable found — semantic search is empty[/skip]")
        return False

    return True


async def run(console: Console, path: str | Path, *, force: bool = False):
    """Phase 1 + Phase 2 together. Returns the ProjectIndex, or None."""
    index = scan(console, path)
    if index is None:
        return None
    describe(console, index)
    console.print()
    await embed(console, index.path, force=force)
    return index
