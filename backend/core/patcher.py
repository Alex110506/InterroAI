"""
Applies surgical search-and-replace patches to files on disk.
"""
from __future__ import annotations

from pathlib import Path


def apply_patch(file_path: str, search_block: str, replace_block: str) -> tuple[bool, str]:
    """
    Replace the first occurrence of search_block in file_path with replace_block.
    Returns (success, error_message).
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"

    content = path.read_text(encoding="utf-8")

    if search_block not in content:
        return False, (
            f"Search block not found verbatim in {file_path}. "
            "Ensure indentation and whitespace match the file exactly."
        )

    patched = content.replace(search_block, replace_block, 1)
    path.write_text(patched, encoding="utf-8")
    return True, ""
