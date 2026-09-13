"""
Startup indexing for the CLI.

Phase 1 (file tree + git context) is cheap and always runs. Phase 2 used to be
skipped whenever the project already had vectors, because it re-embedded the
whole project at full API cost and a CLI is relaunched constantly — which meant
the index silently went stale the moment you edited anything outside the tool.

Phase 2 is now incremental: it hashes the files, embeds only what changed, and
prunes what disappeared. An untouched project therefore costs one local hashing
pass and no API calls, so it is cheap enough to reconcile on every startup and
the index stays honest. `--reindex` / `/index` still forces a full rebuild, for
when the store itself is suspect rather than merely out of date.
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

    A failure here is reported but not fatal: the coder's `search_semantic`
    tool simply returns nothing until it succeeds, and the user may simply
    need to set an API key with `/user` and re-run `/index`.
    """
    failed = False
    summary: dict = {}

    with console.status("[hint]Indexing…[/hint]", spinner="dots") as status:
        async for event in embed_project(path, force=force):
            step = event.get("step")

            if step == "error":
                status.stop()
                console.print(f"  [fail]{FAIL} Indexing failed — {event.get('message')}[/fail]")
                console.print("  [hint]Set an API key with /user, then run /index[/hint]")
                failed = True
                break

            if step == "done":
                summary = event
                break

            if event.get("status") == "progress" and step == "C":
                status.update(
                    f"[hint]Embedding {event.get('embedded')}/{event.get('total')}…[/hint]"
                )
            elif event.get("status") == "done":
                line = _describe_step(step, event)
                if line:
                    status.stop()
                    console.print(f"  [ok]{PASS}[/ok] [hint]{line}[/hint]")
                    status.start()

    if failed:
        return False

    _report_skipped(console, summary)

    if not collection_size(path):
        console.print(f"  [skip]{SKIP} Nothing indexable found — semantic search is empty[/skip]")
        return False

    return True


def _describe_step(step: str, event: dict) -> str:
    """
    One line per finished step, naming what the run actually did.

    "Embedded 0" on an unchanged project looks like a failure, so the reconcile
    counts are what gets reported instead: the interesting number is how much
    work was avoided.
    """
    label = _STEP_LABELS.get(step, step)

    if step == "A":
        parts = [f"{event.get('files', 0)} files"]
        if event.get("changed"):
            parts.append(f"{event['changed']} changed")
        if event.get("unchanged"):
            parts.append(f"{event['unchanged']} unchanged")
        if event.get("removed"):
            parts.append(f"{event['removed']} gone")
        return f"{label} {' · '.join(parts)}"

    if step == "C":
        total = event.get("total", 0)
        cached = event.get("cached", 0)
        return f"{label} {total}" + (f" ({cached} from cache)" if cached else "")

    if step == "D":
        stored = event.get("stored", 0)
        deleted = event.get("deleted", 0)
        if not stored and not deleted:
            return ""
        parts = [f"{stored} chunks"]
        if deleted:
            parts.append(f"{deleted} stale removed")
        return f"{label} {' · '.join(parts)}"

    return f"{label} {event.get('chunks', 0)}"


def _report_skipped(console: Console, summary: dict) -> None:
    """
    Say so when chunks could not be embedded.

    An index missing some of its content must never read as a complete one —
    the same reason `CheckStatus.SKIPPED` exists in `core/sandbox.py`.
    """
    skipped = summary.get("skipped") or 0
    if not skipped:
        return

    files = summary.get("skipped_files") or []
    shown = ", ".join(files[:3]) + (" …" if len(files) > 3 else "")
    console.print(
        f"  [warn]{SKIP} {skipped} chunk(s) could not be embedded and were skipped"
        f"{f' — {shown}' if shown else ''}[/warn]"
    )
    console.print("  [hint]Semantic search will not cover those parts.[/hint]")


async def run(console: Console, path: str | Path, *, force: bool = False):
    """Phase 1 + Phase 2 together. Returns the ProjectIndex, or None."""
    index = scan(console, path)
    if index is None:
        return None
    describe(console, index)
    console.print()
    await embed(console, index.path, force=force)
    return index
