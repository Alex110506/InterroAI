"""
Chunking. The line numbers matter most: they are what the agent is shown when
a semantic search hit is rendered, so an off-by-one sends it to the wrong code.
"""
from __future__ import annotations

import pytest

from core.workspace.chunker import _MAX_FILE_BYTES, chunk_file, is_indexable

# ── is_indexable ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["a.py", "a.js", "a.tsx", "a.rs", "a.go", "a.md", "a.json", "a.yaml", "a.sh", "a.sql"]
)
def test_known_text_and_code_types_are_indexable(tmp_path, name):
    assert is_indexable(tmp_path / name)


@pytest.mark.parametrize("name", ["a.png", "a.zip", "a.pyc", "a.exe", "noextension"])
def test_binary_and_unknown_types_are_not_indexable(tmp_path, name):
    assert not is_indexable(tmp_path / name)


def test_extension_matching_is_case_insensitive(tmp_path):
    assert is_indexable(tmp_path / "A.PY")
    assert is_indexable(tmp_path / "README.MD")


# ── Degenerate inputs ────────────────────────────────────────────────────────


def test_empty_file_produces_no_chunks(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("", encoding="utf-8")
    assert chunk_file(path, tmp_path) == []


def test_whitespace_only_file_produces_no_chunks(tmp_path):
    path = tmp_path / "blank.py"
    path.write_text("\n\n   \n\t\n", encoding="utf-8")
    assert chunk_file(path, tmp_path) == []


def test_oversized_file_is_skipped(tmp_path):
    path = tmp_path / "big.py"
    path.write_text("x = 1\n" * (_MAX_FILE_BYTES // 3), encoding="utf-8")
    assert path.stat().st_size > _MAX_FILE_BYTES
    assert chunk_file(path, tmp_path) == []


def test_unreadable_file_returns_empty_rather_than_raising(tmp_path):
    """Indexing walks whole trees; one bad file must not abort the run."""
    assert chunk_file(tmp_path / "missing.py", tmp_path) == []


def test_undecodable_bytes_do_not_raise(tmp_path):
    path = tmp_path / "weird.py"
    path.write_bytes(b"x = '\xff\xfe'\n")
    chunk_file(path, tmp_path)  # errors="ignore" — must not raise


# ── Chunk shape ──────────────────────────────────────────────────────────────


def test_chunk_carries_the_path_relative_to_the_root(tmp_path):
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    path = nested / "mod.py"
    path.write_text("value = 1\n", encoding="utf-8")
    assert chunk_file(path, tmp_path)[0]["file_path"] == "pkg/sub/mod.py"


def test_every_chunk_has_the_expected_keys(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("a = 1\nb = 2\n", encoding="utf-8")
    chunk = chunk_file(path, tmp_path)[0]
    assert set(chunk) == {"file_path", "content", "start_line", "end_line"}


def test_small_file_is_a_single_chunk_spanning_its_lines(tmp_path):
    path = tmp_path / "small.py"
    path.write_text("one = 1\ntwo = 2\nthree = 3\n", encoding="utf-8")
    chunks = chunk_file(path, tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] >= 3


def test_line_numbers_are_one_based(tmp_path):
    """Editors are 1-indexed; a 0 here would mislead the agent on every hit."""
    path = tmp_path / "m.py"
    path.write_text("first = 1\n", encoding="utf-8")
    assert chunk_file(path, tmp_path)[0]["start_line"] == 1


def test_line_ranges_are_ordered_and_within_the_file(tmp_path):
    path = tmp_path / "long.py"
    body = "".join(f"line_{i} = {i}\n" for i in range(400))
    path.write_text(body, encoding="utf-8")
    total_lines = body.count("\n") + 1

    chunks = chunk_file(path, tmp_path)
    assert len(chunks) > 1, "a 400-line file should split into several chunks"
    for chunk in chunks:
        assert 1 <= chunk["start_line"] <= chunk["end_line"] <= total_lines


def test_reported_start_line_matches_the_chunk_content(tmp_path):
    """The strongest guarantee: seek to start_line and you find the chunk."""
    path = tmp_path / "long.py"
    body = "".join(f"line_{i} = {i}\n" for i in range(300))
    path.write_text(body, encoding="utf-8")
    source_lines = body.splitlines()

    for chunk in chunk_file(path, tmp_path):
        first_content_line = chunk["content"].splitlines()[0].strip()
        if not first_content_line:
            continue
        assert first_content_line in source_lines[chunk["start_line"] - 1]


def test_chunks_that_would_share_a_start_line_are_collapsed(tmp_path):
    """
    A chunk's id is its file and its start line, so two chunks cannot begin on
    the same line. One line longer than the chunk size less the overlap gets cut
    inside itself, and every piece maps back to that one line.
    """
    path = tmp_path / "wide.tsx"
    wide = "const classes = [" + ", ".join(f'"class-{i}"' for i in range(300)) + "]"
    path.write_text(f"import x from 'y'\n{wide}\nexport default x\n", encoding="utf-8")

    chunks = chunk_file(path, tmp_path)

    starts = [chunk["start_line"] for chunk in chunks]
    assert starts, "the file is indexable, so it must produce chunks"
    assert len(starts) == len(set(starts)), "a store cannot hold two chunks under one id"
    assert max(chunk["end_line"] for chunk in chunks) >= 3, "the surviving chunk covers the rest"


def test_no_chunk_is_blank(tmp_path):
    path = tmp_path / "gaps.py"
    path.write_text("a = 1\n\n\n\n\n\nb = 2\n", encoding="utf-8")
    assert all(c["content"].strip() for c in chunk_file(path, tmp_path))


def test_markdown_uses_its_own_splitter_without_error(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\n" + ("prose paragraph. " * 200), encoding="utf-8")
    assert chunk_file(path, tmp_path)


def test_unregistered_extension_falls_back_to_the_generic_splitter(tmp_path):
    path = tmp_path / "conf.toml"
    path.write_text("[section]\nkey = 'value'\n", encoding="utf-8")
    assert chunk_file(path, tmp_path)
