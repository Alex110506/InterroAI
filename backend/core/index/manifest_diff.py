"""
The sync diff, as a pure function.

The local index and the Cloud API's sync endpoint both answer "which of these
files has changed since the index last saw it?" with this one function, so the
two builds cannot drift into different answers.
"""
from __future__ import annotations

from collections.abc import Mapping

from contracts.indexing import FileState, SyncResult


def diff_manifest(
    files: list[FileState],
    indexed: Mapping[str, str],
    *,
    force: bool = False,
) -> SyncResult:
    """
    Compare the client's *files* with *indexed*, which maps each path the index
    holds to the hash of the file its chunks were cut from.

    A file is unchanged only when both hashes exist and match. An empty hash —
    a file the client could not read — therefore always counts as changed, so
    an unreadable file is re-attempted rather than silently trusted.

    With *force*, every file is changed and nothing is removed: a forced job
    resets the index, so there is nothing left in it to prune.
    """
    paths = [f.file_path for f in files]
    if force:
        return SyncResult(changed=paths)

    changed = [
        f.file_path
        for f in files
        if not (f.file_hash and indexed.get(f.file_path) == f.file_hash)
    ]
    return SyncResult(
        changed=changed,
        removed=sorted(set(indexed) - set(paths)),
        unchanged=len(paths) - len(changed),
    )
