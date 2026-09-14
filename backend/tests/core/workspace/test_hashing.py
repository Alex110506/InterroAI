"""
File hashing. Sync and search staleness both compare these for equality, so
the function has to agree with itself exactly.
"""
from __future__ import annotations

import hashlib

import core.workspace.hashing as hashing
from core.workspace.hashing import file_hash


def test_the_hash_is_the_sha256_of_the_bytes(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"x = 1\n")
    assert file_hash(path) == hashlib.sha256(b"x = 1\n").hexdigest()


def test_identical_content_hashes_identically(tmp_path):
    (tmp_path / "a.py").write_text("same\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("same\n", encoding="utf-8")
    assert file_hash(tmp_path / "a.py") == file_hash(tmp_path / "b.py")


def test_a_one_byte_edit_changes_the_hash(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    before = file_hash(path)
    path.write_text("x = 2\n", encoding="utf-8")
    assert file_hash(path) != before


def test_a_file_longer_than_one_read_is_hashed_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(hashing, "_READ_SIZE", 4)
    data = b"abcdefghij" * 3
    path = tmp_path / "long.py"
    path.write_bytes(data)
    assert file_hash(path) == hashlib.sha256(data).hexdigest()


def test_an_unreadable_file_hashes_to_empty(tmp_path):
    """Empty never matches a stored hash, so the file is simply re-attempted."""
    assert file_hash(tmp_path / "missing.py") == ""
