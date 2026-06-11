"""
Project indexing — Hybrid Context Strategy (Section 4 of architecture spec).

Phase 1 (HTTP POST /index): file tree + git context → returns ProjectIndex JSON.
Phase 2 (WebSocket /ws/embed): Steps A-D → .gitignore-aware walk, AST chunking,
  OpenAI text-embedding-3-small, ChromaDB storage with live progress events.
"""

import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.chunker import chunk_file, is_indexable
from core.embeddings import embed_texts
from core.vector_store import store_chunks

try:
    import pathspec as _pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ── Constants ─────────────────────────────────────────────────────────────────

_EXCLUDE_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "out", ".next", ".nuxt", ".svelte-kit",
    "coverage", ".coverage", "htmlcov",
}

_EXCLUDE_FILES: set[str] = {".DS_Store", "Thumbs.db", "desktop.ini"}

_MAX_DEPTH = 6


# ── Pydantic models ───────────────────────────────────────────────────────────

class FileNode(BaseModel):
    name: str
    path: str          # relative to project root
    kind: str          # 'file' | 'dir'
    children: list["FileNode"] | None = None


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


class IndexRequest(BaseModel):
    path: str


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
        if entry.name.startswith(".") and entry.name not in {".env", ".envrc", ".python-version"}:
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
        return result.stdout.strip() if result.returncode == 0 else None
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


# ── Phase 2: .gitignore-aware file walker ─────────────────────────────────────

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
            if name.startswith(".") and name not in {".env", ".envrc", ".python-version"}:
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


# ── Phase 1: HTTP endpoint ────────────────────────────────────────────────────

@router.post("/index", response_model=ProjectIndex)
async def index_project(body: IndexRequest) -> ProjectIndex:
    """
    Index a project folder (Phase 1).

    Scans the file tree and gathers git context so the agent has
    structural awareness of the workspace before any prompt is sent.
    """
    root = Path(body.path).expanduser().resolve()

    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {root}")
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {root}")

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
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Phase 2: WebSocket embedding endpoint ────────────────────────────────────

@router.websocket("/ws/embed")
async def embed_project_ws(websocket: WebSocket) -> None:
    """
    Phase 2 RAG indexing pipeline (Sections 3A-D).

    Client sends:  { "path": "/abs/path/to/project" }
    Server emits a stream of JSON progress events:
      { "step": "A"|"B"|"C"|"D", "status": "start"|"done"|"progress", ...fields }
      { "step": "done" }
      { "step": "error", "message": "..." }
    """
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        root = Path(payload["path"]).expanduser().resolve()
        if not root.is_dir():
            await websocket.send_json({"step": "error", "message": f"Not a directory: {root}"})
            return

        # ── A: .gitignore-aware file traversal ───────────────────────────────
        await websocket.send_json({"step": "A", "status": "start"})
        files = await asyncio.to_thread(_walk_indexable_files, root)
        await websocket.send_json({"step": "A", "status": "done", "files": len(files)})

        # ── B: semantic chunking ─────────────────────────────────────────────
        await websocket.send_json({"step": "B", "status": "start"})
        all_chunks: list[dict] = []
        for path in files:
            all_chunks.extend(chunk_file(path, root))
        await websocket.send_json({"step": "B", "status": "done", "chunks": len(all_chunks)})

        if not all_chunks:
            await websocket.send_json({"step": "done"})
            return

        # ── C: OpenAI text-embedding-3-small ────────────────────────────────
        await websocket.send_json(
            {"step": "C", "status": "start", "total": len(all_chunks)}
        )

        async def _on_progress(embedded: int, total: int) -> None:
            await websocket.send_json(
                {"step": "C", "status": "progress", "embedded": embedded, "total": total}
            )

        all_embeddings = await embed_texts(
            [c["content"] for c in all_chunks],
            on_progress=_on_progress,
        )
        await websocket.send_json({"step": "C", "status": "done", "total": len(all_chunks)})

        # ── D: ChromaDB persistent storage ───────────────────────────────────
        await websocket.send_json({"step": "D", "status": "start"})
        await asyncio.to_thread(store_chunks, str(root), all_chunks, all_embeddings)
        await websocket.send_json(
            {"step": "D", "status": "done", "stored": len(all_chunks)}
        )

        await websocket.send_json({"step": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"step": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
