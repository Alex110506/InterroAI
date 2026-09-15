"""
The sync diff. Pure, so every rule about what "changed" means is tested here
once, for both the local index and the cloud sync endpoint.
"""
from __future__ import annotations

from contracts.indexing import FileState
from core.index.manifest_diff import diff_manifest


def _files(*pairs) -> list[FileState]:
    return [FileState(file_path=path, file_hash=digest) for path, digest in pairs]


def test_a_new_project_has_every_file_changed_and_nothing_removed():
    result = diff_manifest(_files(("a.py", "h1"), ("b.py", "h2")), {})
    assert (result.changed, result.removed, result.unchanged) == (["a.py", "b.py"], [], 0)


def test_a_matching_hash_is_unchanged():
    result = diff_manifest(_files(("a.py", "h1"), ("b.py", "h2")), {"a.py": "h1"})
    assert result.changed == ["b.py"]
    assert result.unchanged == 1


def test_a_differing_hash_is_changed():
    assert diff_manifest(_files(("a.py", "edited")), {"a.py": "h1"}).changed == ["a.py"]


def test_an_indexed_file_the_client_no_longer_has_is_removed():
    result = diff_manifest(_files(("a.py", "h1")), {"a.py": "h1", "z.py": "h", "m.py": "h"})
    assert result.removed == ["m.py", "z.py"]


def test_changed_files_keep_the_order_they_were_sent_in():
    """The client chunks in this order, and batches are cut from chunk order."""
    result = diff_manifest(_files(("z.py", "1"), ("a.py", "2"), ("m.py", "3")), {})
    assert result.changed == ["z.py", "a.py", "m.py"]


def test_an_unreadable_file_is_never_trusted_as_unchanged():
    """An empty hash on both sides still is not a match."""
    assert diff_manifest(_files(("a.py", "")), {"a.py": ""}).changed == ["a.py"]


def test_force_changes_everything_and_removes_nothing():
    """A forced job resets the index, so there is nothing left to prune."""
    result = diff_manifest(_files(("a.py", "h1")), {"a.py": "h1", "gone.py": "h2"}, force=True)
    assert (result.changed, result.removed, result.unchanged) == (["a.py"], [], 0)
