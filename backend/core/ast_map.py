"""
AST-based repo map: extracts class and function signatures from source files.
Gives the Coding Agent global dependency awareness without full file content.
"""
from __future__ import annotations

import ast
import re
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
    for f in sorted(root.rglob("*")):
        if count >= _MAX_FILES:
            sections.append("# ... (truncated — too many files)")
            break
        if not f.is_file() or any(p in _SKIP_DIRS for p in f.parts):
            continue
        suffix = f.suffix.lower()
        if suffix in _PY_EXT:
            body = _map_python(f)
        elif suffix in _JS_EXT:
            body = _map_js(f)
        else:
            continue
        if body:
            sections.append(f"## {f.relative_to(root)}\n{body}")
            count += 1

    return "\n\n".join(sections)
