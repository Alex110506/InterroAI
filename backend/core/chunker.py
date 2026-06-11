"""
Step B — Language-Aware Chunking (Section 3B of architecture spec).

Uses LangChain's RecursiveCharacterTextSplitter with language-specific
separator sets (chunk_size=800, chunk_overlap=200). Falls back to the
generic splitter for file types without a registered language profile.

Each chunk dict: {file_path, content, start_line, end_line}
"""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# ── File type sets ────────────────────────────────────────────────────────────

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs",
    ".html", ".css", ".scss", ".sass", ".sql",
})

_TEXT_EXTENSIONS = _CODE_EXTENSIONS | frozenset({
    ".md", ".mdx", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env",
    ".sh", ".bash", ".zsh", ".fish",
})

# ── Tuning constants ──────────────────────────────────────────────────────────

_MAX_FILE_BYTES = 200_000  # skip files > 200 KB
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 200

# ── Extension → Language map ──────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, Language] = {
    ".py":  Language.PYTHON,
    ".js":  Language.JS,  ".jsx": Language.JS,
    ".mjs": Language.JS,  ".cjs": Language.JS,
    ".ts":  Language.TS,  ".tsx": Language.TS,
    ".go":  Language.GO,
    ".rs":  Language.RUST,
    ".java": Language.JAVA,
    ".kt":  Language.KOTLIN,
    ".swift": Language.SWIFT,
    ".rb":  Language.RUBY,
    ".php": Language.PHP,
    ".c":   Language.C,   ".h":   Language.C,
    ".cpp": Language.CPP, ".cc":  Language.CPP, ".hpp": Language.CPP,
    ".cs":  Language.CSHARP,
    ".md":  Language.MARKDOWN, ".mdx": Language.MARKDOWN,
    ".rst": Language.RST,
    ".html": Language.HTML,
}


# ── Splitter factory ──────────────────────────────────────────────────────────

def _splitter(ext: str) -> RecursiveCharacterTextSplitter:
    lang = _EXT_TO_LANG.get(ext)
    if lang is not None:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                lang,
                chunk_size=_CHUNK_SIZE,
                chunk_overlap=_CHUNK_OVERLAP,
                add_start_index=True,
            )
        except (ValueError, KeyError):
            pass  # unknown lang in this version — use generic
    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        add_start_index=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def is_indexable(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTENSIONS


def chunk_file(path: Path, root: Path) -> list[dict]:
    """
    Read *path* and return a list of chunk dicts:
      {file_path, content, start_line, end_line}

    Returns [] if the file is too large, unreadable, or produces no content.
    """
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return []
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not source.strip():
        return []

    rel = str(path.relative_to(root))
    ext = path.suffix.lower()
    docs = _splitter(ext).create_documents([source])

    result = []
    for doc in docs:
        content = doc.page_content
        if not content.strip():
            continue
        start_char: int = doc.metadata.get("start_index", 0)
        end_char = start_char + len(content)
        start_line = source[:start_char].count("\n") + 1
        end_line = source[:end_char].count("\n") + 1
        result.append({
            "file_path": rel,
            "content": content,
            "start_line": start_line,
            "end_line": end_line,
        })

    return result
