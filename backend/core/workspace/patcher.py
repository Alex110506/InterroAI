"""
Applies surgical search-and-replace patches to files on disk.
"""
from __future__ import annotations

from pathlib import Path

from core.errors import FileNotFoundInProjectError, PatchError


def apply_patch(file_path: str, search_block: str, replace_block: str) -> None:
    """
    Replace the single occurrence of *search_block* in *file_path*.

    Returns None on success. Every failure mode is a typed `ToolError` whose
    message is written to be read by the model, so the agent can hand it back
    verbatim and let the model correct itself.

    Raises:
        FileNotFoundInProjectError: the file does not exist.
        PatchError: the search block is missing, or appears more than once.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundInProjectError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    occurrences = content.count(search_block)
    if occurrences == 0:
        raise PatchError(
            f"Search block not found verbatim in {file_path}. "
            "Ensure indentation and whitespace match the file exactly."
        )
    # Patching the first of several identical blocks is a coin flip — it may
    # silently edit the wrong one. Make the model disambiguate instead.
    if occurrences > 1:
        raise PatchError(
            f"Search block appears {occurrences} times in {file_path}; "
            "refusing to guess which one you meant. Include more surrounding "
            "context so the search block matches exactly one location."
        )

    path.write_text(content.replace(search_block, replace_block, 1), encoding="utf-8")
