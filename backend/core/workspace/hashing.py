"""
File content hashing, shared by the two places that must agree on it.

The hash a sync sends for a file and the hash a search result is checked
against are compared for equality, so both have to come from the same function
— a difference in chunk size or encoding would make every file look changed,
and every search hit look stale.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_READ_SIZE = 1 << 20


def file_hash(path: Path) -> str:
    """sha256 of a file's bytes, or "" if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(_READ_SIZE):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()
