"""
Project indexing — the runtime's half (Section 4 of the architecture spec).

Transport-agnostic. Two phases, both driven by the HTTP/WebSocket adapters in
`api/projects.py`:

  Phase 1  `build_index(path)`   — file tree + git context, returned in one shot.
  Phase 2  `embed_project(path)` — an async generator of A/B/C/D progress events
                                   covering traversal → chunking → embedding →
                                   storage.

Phase 2 is split down the middle, along the line the cloud build will cut it:

  * **Here, on the machine that has the files:** walk the project, hash every
    file, ask the `SemanticIndex` which of them changed (step A), and chunk only
    those (step B).
  * **In the index's worker (`core/index/indexer.py`):** embed the chunks, store them,
    and prune whatever they replaced (steps C and D).

The halves meet only through `contracts.indexing` — a `SyncRequest`, a
`ChunkUpload`, and the `IndexEvent`s relayed back — so this module never touches
the vector store or an embedding model, and `tests/test_boundaries.py` fails if
it starts to.

Phase 2 is also **incremental**, which is what makes it cheap enough to run
every time a project is opened. A project nobody has touched costs one local
hashing pass and one sync, and produces no job at all. `force=True` skips
reconciliation and rebuilds the project's index from scratch.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from contracts.indexing import Chunk, ChunkUpload, FileState, SyncRequest
from core import providers
from core.errors import InterroAIError
from core.index.semantic_index import SemanticIndex
from core.workspace.chunker import chunk_file, is_indexable
from core.workspace.hashing import file_hash

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


def _manifest(root: Path, files: list[Path]) -> list[FileState]:
    """What the client has: every indexable file, and the hash of its bytes."""
    return [
        FileState(file_path=str(path.relative_to(root)), file_hash=file_hash(path))
        for path in files
    ]


def _chunk_changed(root: Path, changed: list[str], hashes: dict[str, str]) -> list[Chunk]:
    """Cut the changed files into chunks, each stamped with its file's hash."""
    chunks: list[Chunk] = []
    for relative in changed:
        path = root / relative
        # Hashed during the scan already; hashed again only if that read
        # failed, so a file that has since become readable records a real hash.
        digest = hashes.get(relative) or file_hash(path)
        chunks.extend(Chunk(file_hash=digest, **piece) for piece in chunk_file(path, root))
    return chunks


async def _embed_steps(root: Path, *, force: bool, index: SemanticIndex) -> AsyncIterator[dict]:
    """The A-D pipeline proper. Raises; `embed_project` turns that into an event."""
    project_id = str(root)

    # ── A: traversal and hashing, then a sync against what the index holds ───
    yield {"step": "A", "status": "start"}
    files = await asyncio.to_thread(_walk_indexable_files, root)
    manifest = await asyncio.to_thread(_manifest, root, files)
    sync = await index.sync(SyncRequest(project_id=project_id, files=manifest, force=force))
    yield {
        "step": "A",
        "status": "done",
        "files": len(files),
        "changed": len(sync.changed),
        "removed": len(sync.removed),
        "unchanged": sync.unchanged,
    }

    # ── B: semantic chunking, of the changed files only ──────────────────────
    yield {"step": "B", "status": "start"}
    hashes = {entry.file_path: entry.file_hash for entry in manifest}
    chunks = await asyncio.to_thread(_chunk_changed, root, sync.changed, hashes)
    yield {"step": "B", "status": "done", "chunks": len(chunks)}

    if not (sync.changed or sync.removed or force):
        # Nothing on disk moved, so there is no job: no upload, no queue
        # message, and — in the cloud — no worker woken up from zero replicas.
        yield {"step": "D", "status": "done", "stored": 0, "deleted": 0}
        yield {
            "step": "done",
            "embedded": 0,
            "cached": 0,
            "skipped": 0,
            "deleted": 0,
            "unchanged": sync.unchanged,
        }
        return

    # ── C, D: the worker's half, relayed as it reports ───────────────────────
    job_id = await index.upload(
        ChunkUpload(
            project_id=project_id,
            chunks=chunks,
            changed_paths=sync.changed,
            removed_paths=sync.removed,
            reset=force,
        )
    )

    finished = False
    async for event in index.job_events(job_id):
        wire = event.to_wire()
        if event.step == "done":
            # Only this side knows how many files the sync let it skip.
            wire["unchanged"] = sync.unchanged
        finished = finished or event.step in {"done", "error"}
        yield wire

    if not finished:
        # Every consumer stops on `done` or `error`. A stream that ends without
        # either would otherwise leave the UI waiting on a job that is over.
        yield {"step": "error", "message": "The indexing job ended without reporting a result."}


async def embed_project(
    path: str | Path,
    *,
    force: bool = False,
    index: SemanticIndex | None = None,
) -> AsyncIterator[dict]:
    """
    Phase 2 RAG indexing pipeline (Sections 3A-D).

    Reconciles against what the index already holds and embeds only what
    changed. Pass *force* to discard the project's index and rebuild it. Pass
    *index* to use a specific `SemanticIndex`; by default `core.providers`
    chooses.

    Yields progress events:
      {"step": "A"|"B"|"C"|"D", "status": "start"|"done"|"progress", ...}
      {"step": "done", "embedded", "cached", "skipped", "deleted", "unchanged"}
      {"step": "error", "message": "..."}

    Failures on either half arrive as an error event rather than an exception.
    Every caller has to render them anyway, so an exception handler at each call
    site would have nothing to add. Chunks embedded before the failure are
    already stored — the next run reconciles and picks up the rest.
    """
    try:
        root = Path(path).expanduser().resolve()
    except OSError as exc:
        yield {"step": "error", "message": str(exc)}
        return

    if not root.is_dir():
        yield {"step": "error", "message": f"Not a directory: {root}"}
        return

    index = index or providers.semantic_index()
    try:
        async for event in _embed_steps(root, force=force, index=index):
            yield event
    except InterroAIError as exc:
        # Expected (e.g. no API key) — the message is already user-facing.
        logger.info("Indexing stopped: %s", exc)
        event = {"step": "error", "message": str(exc)}
        if exc.code:
            event["code"] = exc.code
        yield event
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during project indexing")
        yield {"step": "error", "message": str(exc)}
