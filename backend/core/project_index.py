"""
Project indexing — Hybrid Context Strategy (Section 4 of the architecture spec).

Transport-agnostic. Two phases, both usable from the CLI directly and from the
HTTP/WebSocket adapters in `api/projects.py`:

  Phase 1  `build_index(path)`   — file tree + git context, returned in one shot.
  Phase 2  `embed_project(path)` — an async generator of A/B/C/D progress events
                                   covering traversal → chunking → embedding →
                                   ChromaDB storage.

Phase 2 is a generator rather than a callback-driven routine so that a caller
can render progress however it likes — as WebSocket frames, or as a live
terminal progress display — without this module knowing which.

Phase 2 is also **incremental**, which is what makes it cheap enough to run on
every startup. Each stored chunk carries the hash of the file it came from, so
a run begins by comparing the hashes on disk with the hashes in the store and
splits the project three ways:

  * unchanged files — nothing to do, and nothing to pay for;
  * changed or new files — re-chunked and re-embedded;
  * files that are gone, and chunks a shrunken file no longer produces —
    deleted, because an orphaned chunk still answers searches and attributes
    content to a file that does not contain it any more.

A project nobody has touched therefore costs one local hashing pass and zero
API calls, and the index stops drifting out of date between explicit re-indexes.
`force=True` skips reconciliation and rebuilds from scratch — see
`vector_store.reset_collection`.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from core.chunker import chunk_file, is_indexable
from core.embeddings import embed_batches
from core.errors import InterroAIError
from core.vector_store import (
    chunk_id,
    delete_ids,
    reset_collection,
    store_chunks,
    stored_manifest,
)

logger = logging.getLogger(__name__)

try:
    import pathspec as _pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False


# ── Constants ─────────────────────────────────────────────────────────────────

_EXCLUDE_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "out", ".next", ".nuxt", ".svelte-kit",
    "coverage", ".coverage", "htmlcov",
}

_EXCLUDE_FILES: set[str] = {".DS_Store", "Thumbs.db", "desktop.ini"}

_MAX_DEPTH = 6

#: Dotted entries worth indexing despite the blanket "skip hidden" rule.
_KEEP_HIDDEN = {".env", ".envrc", ".python-version"}


# ── Pydantic models ───────────────────────────────────────────────────────────

class FileNode(BaseModel):
    name: str
    path: str          # relative to project root
    kind: str          # 'file' | 'dir'
    children: list[FileNode] | None = None


class GitContext(BaseModel):
    is_git_repo: bool
    branch: str | None = None
    modified_files: list[str] = []
    untracked_files: list[str] = []
    recent_commits: list[str] = []


class ProjectIndex(BaseModel):
    path: str
    name: str
    file_tree: FileNode
    git_context: GitContext
    total_files: int
    total_dirs: int
    languages: list[str]        # detected from file extensions
    indexed_at: str


class ProjectPathError(ValueError):
    """The requested path is missing or is not a directory."""


# ── File tree builder ─────────────────────────────────────────────────────────

def _build_tree(
    root: Path,
    rel_root: Path,
    depth: int = 0,
) -> tuple[FileNode, int, int]:
    """
    Recursively build a FileNode tree.
    Returns (node, file_count, dir_count).
    """
    node = FileNode(
        name=root.name,
        path=str(root.relative_to(rel_root)),
        kind="dir",
        children=[],
    )
    total_files = 0
    total_dirs = 1

    if depth >= _MAX_DEPTH:
        return node, total_files, total_dirs

    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return node, total_files, total_dirs

    for entry in entries:
        if entry.name.startswith(".") and entry.name not in _KEEP_HIDDEN:
            continue
        if entry.name in _EXCLUDE_DIRS or entry.name in _EXCLUDE_FILES:
            continue
        if entry.is_symlink():
            continue

        if entry.is_dir():
            child, fc, dc = _build_tree(entry, rel_root, depth + 1)
            node.children.append(child)
            total_files += fc
            total_dirs += dc
        elif entry.is_file():
            node.children.append(
                FileNode(name=entry.name, path=str(entry.relative_to(rel_root)), kind="file")
            )
            total_files += 1

    return node, total_files, total_dirs


# ── Language detection ────────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin",
    ".swift": "Swift", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".cpp": "C++", ".h": "C/C++",
    ".cs": "C#", ".fs": "F#",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS",
    ".sql": "SQL", ".sh": "Shell", ".bash": "Shell",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON",
    ".md": "Markdown",
}


def _detect_languages(tree: FileNode) -> list[str]:
    """Walk the tree and collect the unique languages present."""
    counts: dict[str, int] = {}

    def _walk(node: FileNode) -> None:
        if node.kind == "file":
            ext = "." + node.name.rsplit(".", 1)[-1].lower() if "." in node.name else ""
            lang = _EXT_TO_LANG.get(ext)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        for child in node.children or []:
            _walk(child)

    _walk(tree)
    return [lang for lang, _ in sorted(counts.items(), key=lambda x: -x[1])]


# ── Git context ───────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd,
            capture_output=True, text=True, timeout=5,
        )
        # Only trailing newlines. A bare .strip() would also eat the leading
        # space of `git status --porcelain`'s first line (" M file.py"), and
        # the XY-prefix parsing below is column-based — that shifted the first
        # modified filename by one character ("file.py" -> "ile.py").
        return result.stdout.strip("\n") if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_git_context(root: Path) -> GitContext:
    if _run_git(["rev-parse", "--git-dir"], root) is None:
        return GitContext(is_git_repo=False)

    branch = _run_git(["symbolic-ref", "--short", "HEAD"], root)

    modified, untracked = [], []
    status_out = _run_git(["status", "--porcelain"], root) or ""
    for line in status_out.splitlines():
        if len(line) < 3:
            continue
        xy, path = line[:2], line[3:]
        if xy.strip() and xy != "??":
            modified.append(path)
        elif xy == "??":
            untracked.append(path)

    log_out = _run_git(["log", "--oneline", "-5"], root) or ""
    recent_commits = log_out.splitlines()

    return GitContext(
        is_git_repo=True,
        branch=branch,
        modified_files=modified[:20],
        untracked_files=untracked[:10],
        recent_commits=recent_commits,
    )


# ── .gitignore-aware file walker ─────────────────────────────────────────────

def _walk_indexable_files(root: Path) -> list[Path]:
    """
    Walk *root* and return all indexable files, skipping excluded dirs,
    hidden dirs/files, and anything matched by the root .gitignore.
    """
    spec = None
    if _HAS_PATHSPEC:
        gi_path = root / ".gitignore"
        if gi_path.exists():
            spec = _pathspec.PathSpec.from_lines(
                "gitwildmatch",
                gi_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
            )

    files: list[Path] = []

    def _recurse(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            name = entry.name
            if name.startswith(".") and name not in _KEEP_HIDDEN:
                continue
            if entry.is_symlink():
                continue

            rel = entry.relative_to(root)

            if entry.is_dir():
                if name in _EXCLUDE_DIRS:
                    continue
                if spec and spec.match_file(str(rel) + "/"):
                    continue
                _recurse(entry)
            elif entry.is_file():
                if not is_indexable(entry):
                    continue
                if spec and spec.match_file(str(rel)):
                    continue
                files.append(entry)

    _recurse(root)
    return files


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def resolve_project_root(path: str | Path) -> Path:
    """
    Normalise *path* to an existing directory.

    Raises:
        ProjectPathError: the path is missing or is not a directory. Both are
            user input mistakes, not bugs, so callers should show the message.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ProjectPathError(f"Path not found: {root}")
    if not root.is_dir():
        raise ProjectPathError(f"Path is not a directory: {root}")
    return root


def build_index(path: str | Path) -> ProjectIndex:
    """
    Phase 1: scan the file tree and gather git context so the agent has
    structural awareness of the workspace before any prompt is sent.
    """
    root = resolve_project_root(path)

    file_tree, total_files, total_dirs = _build_tree(root, root)
    git_context = _get_git_context(root)
    languages = _detect_languages(file_tree)

    return ProjectIndex(
        path=str(root),
        name=root.name,
        file_tree=file_tree,
        git_context=git_context,
        total_files=total_files,
        total_dirs=total_dirs,
        languages=languages,
        indexed_at=datetime.now(UTC).isoformat(),
    )


# ── Phase 2 ───────────────────────────────────────────────────────────────────

_HASH_READ_SIZE = 1 << 20


def _file_hash(path: Path) -> str:
    """sha256 of a file's bytes, or "" if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(_HASH_READ_SIZE):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


@dataclass(frozen=True)
class _Plan:
    """What a run has to do, decided before a single API call is made."""

    #: Files whose content differs from what the store holds, with their hashes.
    changed: dict[Path, str] = field(default_factory=dict)
    #: Files the store knows and disk no longer has.
    removed: tuple[str, ...] = ()
    #: Chunk ids belonging to those vanished files.
    orphan_ids: tuple[str, ...] = ()
    #: Ids currently stored for each *changed* file, so the ids its new chunks
    #: do not reproduce can be pruned without reading the store a second time.
    stored_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unchanged: int = 0


def _plan_run(root: Path, files: list[Path]) -> _Plan:
    """
    Diff the files on disk against the manifest in the store.

    Hashing is local and cheap next to an embedding call, so this is the step
    that decides what the run costs. It is also the only read of the store per
    run: everything the pruning step needs is carried out of here, which keeps
    the decision consistent even if the files move underneath us afterwards.
    """
    stored = stored_manifest(str(root))

    changed: dict[Path, str] = {}
    stored_ids: dict[str, tuple[str, ...]] = {}
    unchanged = 0
    seen: set[str] = set()

    for path in files:
        relative = str(path.relative_to(root))
        seen.add(relative)
        digest = _file_hash(path)
        known = stored.get(relative)
        if known is not None and digest and known.file_hash == digest:
            unchanged += 1
            continue
        changed[path] = digest
        if known is not None:
            stored_ids[relative] = known.ids

    removed = tuple(sorted(set(stored) - seen))
    orphan_ids = tuple(
        chunk_key for path in removed for chunk_key in stored[path].ids
    )

    return _Plan(
        changed=changed,
        removed=removed,
        orphan_ids=orphan_ids,
        stored_ids=stored_ids,
        unchanged=unchanged,
    )


def _superseded_ids(plan: _Plan, chunks: list[dict]) -> list[str]:
    """
    Ids the re-chunked files used to have and no longer produce.

    A file that shrank is the case that matters: its surviving chunks are
    overwritten by the upsert, but the ones past its new end would otherwise
    stay behind and keep answering searches.
    """
    fresh_ids = {chunk_id(c) for c in chunks}
    return [
        chunk_key
        for path in {c["file_path"] for c in chunks}
        for chunk_key in plan.stored_ids.get(path, ())
        if chunk_key not in fresh_ids
    ]


async def _embed_steps(root: Path, *, force: bool) -> AsyncIterator[dict]:
    """The A-D pipeline proper. Raises; `embed_project` turns that into an event."""
    # ── A: traversal, then reconcile against what is already stored ──────────
    yield {"step": "A", "status": "start"}
    files = await asyncio.to_thread(_walk_indexable_files, root)

    if force:
        # An explicit rebuild: the existing collection is the thing being
        # discarded, so there is nothing to reconcile against.
        await asyncio.to_thread(reset_collection, str(root))
        plan = _Plan(changed={path: "" for path in files})
    else:
        plan = await asyncio.to_thread(_plan_run, root, files)

    yield {
        "step": "A",
        "status": "done",
        "files": len(files),
        "changed": len(plan.changed),
        "removed": len(plan.removed),
        "unchanged": plan.unchanged,
    }

    # ── B: semantic chunking, of the changed files only ──────────────────────
    yield {"step": "B", "status": "start"}
    all_chunks: list[dict] = []
    for file_path, digest in plan.changed.items():
        for chunk in chunk_file(file_path, root):
            # Carried into the store's metadata, so the next run can tell
            # whether this file still matches what was embedded.
            chunk["file_hash"] = digest or _file_hash(file_path)
            all_chunks.append(chunk)
    yield {"step": "B", "status": "done", "chunks": len(all_chunks)}

    stale_ids = list(plan.orphan_ids)
    if all_chunks and not force:
        stale_ids += _superseded_ids(plan, all_chunks)

    if not all_chunks:
        # Nothing to embed. Orphans still have to go, or deleting a file would
        # never actually remove it from search results.
        deleted = await asyncio.to_thread(delete_ids, str(root), stale_ids)
        yield {"step": "D", "status": "done", "stored": 0, "deleted": deleted}
        yield {
            "step": "done",
            "embedded": 0,
            "cached": 0,
            "skipped": 0,
            "deleted": deleted,
            "unchanged": plan.unchanged,
        }
        return

    # ── C: OpenAI text-embedding-3-small, persisted batch by batch ───────────
    yield {"step": "C", "status": "start", "total": len(all_chunks)}

    stored_count = 0
    cached_count = 0
    skipped: list[str] = []

    async for batch in embed_batches([c["content"] for c in all_chunks]):
        if batch.indices:
            # Stored as each batch lands rather than once at the end, so a
            # failure later in the run cannot discard what is already paid for.
            batch_chunks = [all_chunks[i] for i in batch.indices]
            await asyncio.to_thread(store_chunks, str(root), batch_chunks, batch.vectors)
            stored_count += len(batch_chunks)
            cached_count += batch.from_cache

        for position in batch.failures:
            # Its file's other chunks still store the current hash, so the
            # next run sees the file as unchanged and does not re-attempt this
            # one. Deliberate: an item error is deterministic (over the token
            # limit, refused content), so retrying every run would re-pay for
            # the chunks that *do* work to fail identically on this one. The
            # summary reports it, and `--reindex` retries everything.
            skipped.append(all_chunks[position]["file_path"])

        yield {
            "step": "C",
            "status": "progress",
            "embedded": stored_count,
            "total": len(all_chunks),
            "cached": cached_count,
            "skipped": len(skipped),
        }

    yield {
        "step": "C",
        "status": "done",
        "total": len(all_chunks),
        "cached": cached_count,
        "skipped": len(skipped),
    }

    # ── D: prune what the new chunks replaced or outlived ────────────────────
    #
    # After storing, not before: a run that dies mid-way then leaves the old
    # vectors in place rather than a hole, and the next run re-detects the
    # same work from the manifest either way.
    yield {"step": "D", "status": "start"}
    deleted = await asyncio.to_thread(delete_ids, str(root), stale_ids)
    yield {"step": "D", "status": "done", "stored": stored_count, "deleted": deleted}

    yield {
        "step": "done",
        "embedded": stored_count,
        "cached": cached_count,
        "skipped": len(skipped),
        "skipped_files": sorted(set(skipped)),
        "deleted": deleted,
        "unchanged": plan.unchanged,
    }


async def embed_project(path: str | Path, *, force: bool = False) -> AsyncIterator[dict]:
    """
    Phase 2 RAG indexing pipeline (Sections 3A-D).

    Reconciles against what is already stored and embeds only what changed.
    Pass *force* to discard the project's collection and rebuild it instead.

    Yields progress events:
      {"step": "A"|"B"|"C"|"D", "status": "start"|"done"|"progress", ...}
      {"step": "done", "embedded", "cached", "skipped", "deleted", "unchanged"}
      {"step": "error", "message": "..."}

    Failures arrive as an error event rather than an exception. Every caller
    has to render them anyway, so an exception handler at each call site would
    have nothing to add. Chunks embedded before the failure are already
    stored — the next run reconciles and picks up the rest.
    """
    try:
        root = Path(path).expanduser().resolve()
    except OSError as exc:
        yield {"step": "error", "message": str(exc)}
        return

    if not root.is_dir():
        yield {"step": "error", "message": f"Not a directory: {root}"}
        return

    try:
        async for event in _embed_steps(root, force=force):
            yield event
    except InterroAIError as exc:
        # Expected (e.g. no API key) — the message is already user-facing.
        logger.info("Indexing stopped: %s", exc)
        yield {"step": "error", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during project indexing")
        yield {"step": "error", "message": str(exc)}
