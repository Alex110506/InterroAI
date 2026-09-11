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
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from core.chunker import chunk_file, is_indexable
from core.embeddings import embed_texts
from core.errors import InterroAIError
from core.vector_store import store_chunks

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

#: Queue sentinel marking the end of the embedding task. A plain object()
#: cannot collide with a progress event the way a string key could.
_EMBED_FINISHED = object()


async def _embed_with_progress(
    texts: list[str],
    queue: asyncio.Queue,
) -> list[list[float]]:
    """
    Run `embed_texts`, pushing each progress tick onto *queue*.

    `embed_texts` reports progress through a callback, but the caller here is a
    generator and a callback cannot yield on its behalf. The queue bridges the
    two: the callback pushes, the generator drains.
    """

    async def _on_progress(embedded: int, total: int) -> None:
        await queue.put(
            {"step": "C", "status": "progress", "embedded": embedded, "total": total}
        )

    try:
        result = await embed_texts(texts, on_progress=_on_progress)
    except BaseException as exc:  # noqa: BLE001 — re-raised in the consumer
        await queue.put((_EMBED_FINISHED, None, exc))
        raise
    await queue.put((_EMBED_FINISHED, result, None))
    return result


async def _embed_steps(root: Path) -> AsyncIterator[dict]:
    """The A-D pipeline proper. Raises; `embed_project` turns that into an event."""
    # ── A: .gitignore-aware file traversal ───────────────────────────────────
    yield {"step": "A", "status": "start"}
    files = await asyncio.to_thread(_walk_indexable_files, root)
    yield {"step": "A", "status": "done", "files": len(files)}

    # ── B: semantic chunking ─────────────────────────────────────────────────
    yield {"step": "B", "status": "start"}
    all_chunks: list[dict] = []
    for file_path in files:
        all_chunks.extend(chunk_file(file_path, root))
    yield {"step": "B", "status": "done", "chunks": len(all_chunks)}

    if not all_chunks:
        yield {"step": "done"}
        return

    # ── C: OpenAI text-embedding-3-small ─────────────────────────────────────
    yield {"step": "C", "status": "start", "total": len(all_chunks)}

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        _embed_with_progress([c["content"] for c in all_chunks], queue)
    )

    all_embeddings: list[list[float]] = []
    try:
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item and item[0] is _EMBED_FINISHED:
                _, result, exc = item
                if exc is not None:
                    raise exc
                all_embeddings = result
                break
            yield item
    finally:
        # A consumer that stops iterating early (client disconnect, Ctrl-C)
        # must not leave the embedding task running against a queue nobody
        # drains any more.
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    yield {"step": "C", "status": "done", "total": len(all_chunks)}

    # ── D: ChromaDB persistent storage ───────────────────────────────────────
    yield {"step": "D", "status": "start"}
    await asyncio.to_thread(store_chunks, str(root), all_chunks, all_embeddings)
    yield {"step": "D", "status": "done", "stored": len(all_chunks)}

    yield {"step": "done"}


async def embed_project(path: str | Path) -> AsyncIterator[dict]:
    """
    Phase 2 RAG indexing pipeline (Sections 3A-D).

    Yields progress events:
      {"step": "A"|"B"|"C"|"D", "status": "start"|"done"|"progress", ...}
      {"step": "done"}
      {"step": "error", "message": "..."}

    Failures arrive as an error event rather than an exception. Every caller
    has to render them anyway, and there is no partial index to resume from,
    so an exception handler at each call site would have nothing to add.
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
        async for event in _embed_steps(root):
            yield event
    except InterroAIError as exc:
        # Expected (e.g. no API key) — the message is already user-facing.
        logger.info("Indexing stopped: %s", exc)
        yield {"step": "error", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during project indexing")
        yield {"step": "error", "message": str(exc)}
