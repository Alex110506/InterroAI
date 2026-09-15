"""The small pieces of `core/index/ports.py` every adapter relies on."""
from __future__ import annotations

from core.index.ports import IndexedChunk, chunk_id, content_digest, split_chunk_id


def test_a_chunk_id_is_its_file_and_start_line():
    assert chunk_id("src/app.py", 12) == "src/app.py:12"


def test_splitting_an_id_inverts_it():
    assert split_chunk_id(chunk_id("src/app.py", 12)) == ("src/app.py", 12)


def test_a_path_containing_colons_still_splits_correctly():
    """Only the line number is guaranteed to be colon-free."""
    assert split_chunk_id(chunk_id("c:/weird:dir/a.py", 3)) == ("c:/weird:dir/a.py", 3)


def test_an_indexed_chunk_knows_its_id():
    chunk = IndexedChunk("a.py", 7, 20, "h", [0.1])
    assert chunk.id == chunk_id("a.py", 7)


def test_the_content_digest_depends_on_the_text_alone():
    assert content_digest("same text") == content_digest("same text")
    assert content_digest("same text") != content_digest("other text")
