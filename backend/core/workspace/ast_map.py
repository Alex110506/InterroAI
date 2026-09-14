"""
AST-based repo map: extracts class and function signatures from source files.
Gives the Coding Agent global dependency awareness without full file content.

Building the map parses every source file in the project, and the agent needs it
on every request — so `fingerprint_sources` exists to let a caller cache the
result and recompute only when the files behind it actually change. Both walk
the same `_source_files` generator: a fingerprint taken over a different set of
files than the map was built from would validate a stale cache entry.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
_PY_EXT = {".py"}
_JS_EXT = {".js", ".jsx", ".ts", ".tsx"}
_MAX_FILES = 80

_JS_CLASS_RE = re.compile(r"(?:export\s+)?class\s+(\w+)", re.MULTILINE)
_JS_FUNC_RE = re.compile(
    r"(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)\s*\("
    r"|(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(",
    re.MULTILINE,
)


def _py_args(args: ast.arguments) -> str:
    parts: list[str] = []
    for arg in args.posonlyargs:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")
    if args.posonlyargs:
        parts.append("/")
    for arg in args.args:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for arg in args.kwonlyargs:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _map_python(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return ""

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}:")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = _py_args(item.args)
                    ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                    pre = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    lines.append(f"    {pre}def {item.name}({a}){ret}: ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = _py_args(node.args)
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            pre = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            lines.append(f"{pre}def {node.name}({a}){ret}: ...")
    return "\n".join(lines)


def _map_js(path: Path) -> str:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines: list[str] = []
    for m in _JS_CLASS_RE.finditer(src):
        lines.append(f"class {m.group(1)} {{ }}")
    for m in _JS_FUNC_RE.finditer(src):
        name = m.group(1) or m.group(2)
        if name:
            lines.append(f"function {name}() {{ }}")
    return "\n".join(lines)


def _source_files(root: Path) -> Iterator[Path]:
    """Every Python or JS/TS file the map is built from, in a stable order."""
    for f in sorted(root.rglob("*")):
        if not f.is_file() or any(p in _SKIP_DIRS for p in f.parts):
            continue
        if f.suffix.lower() in _PY_EXT | _JS_EXT:
            yield f


def build_repo_map(project_path: str) -> str:
    """Return a compact signature-only map of all Python and JS/TS files."""
    # Resolve so emitted paths align with CoderAgent._path (also resolved),
    # which keeps read_file() lookups consistent on symlinked roots
    # (e.g. macOS /var → /private/var, /tmp → /private/tmp).
    root = Path(project_path).resolve()
    if not root.is_dir():
        return ""

    sections: list[str] = []
    count = 0
    for f in _source_files(root):
        if count >= _MAX_FILES:
            sections.append("# ... (truncated — too many files)")
            break
        body = _map_python(f) if f.suffix.lower() in _PY_EXT else _map_js(f)
        if body:
            sections.append(f"## {f.relative_to(root)}\n{body}")
            count += 1

    return "\n\n".join(sections)


def fingerprint_sources(project_path: str) -> str:
    """
    A cheap identity for the project state `build_repo_map` would read.

    Stat-ing files is orders of magnitude cheaper than parsing them, so a
    caller can check this on every request and only rebuild when it changes.
    Size and mtime rather than content hashes: this decides whether to redo
    local work, and reading every file to avoid re-reading every file would
    defeat the purpose.

    Returns "" for a path that is not a directory, matching `build_repo_map`,
    so an unusable project is never cached against a meaningful key.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return ""

    digest = hashlib.sha256()
    for f in _source_files(root):
        try:
            stat = f.stat()
        except OSError:
            continue
        digest.update(f"{f.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return digest.hexdigest()
