"""
ChromaDB persistence. The `isolated_chroma` fixture redirects the store into a
tmp directory so no test touches `~/.interroai/chroma`.
"""
from __future__ import annotations

import re

import pytest

from core.vector_store import _collection_name, search_chunks, store_chunks

DIM = 8


def _vec(seed: float) -> list[float]:
    return [seed] * DIM


def _chunk(path="main.py", start=1, end=5, content="body"):
    return {"file_path": path, "start_line": start, "end_line": end, "content": content}


# ── Collection naming ────────────────────────────────────────────────────────


def test_path_separators_are_sanitised():
    name = _collection_name("/Users/alex/dev/proj")
    assert "/" not in name


def test_name_stays_within_chroma_length_limits():
    name = _collection_name("/very/deep/" + "segment/" * 40)
    assert 3 <= len(name) <= 63


def test_short_paths_are_padded_to_the_minimum():
    assert len(_collection_name("/a")) >= 3


@pytest.mark.parametrize("path", ["/a", "/p", "/ab", "/x/", "/1"])
def test_generated_names_satisfy_chromas_grammar(path):
    """
    Regression: padding used to append "_", but Chroma requires the name to
    *end* with an alphanumeric, so any project with a 1-2 character path
    crashed on its first index.
    """
    name = _collection_name(path)
    assert 3 <= len(name) <= 63
    assert name[0].isalnum() and name[-1].isalnum(), name
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name), name


def test_truncation_never_leaves_a_trailing_separator():
    """A 63-character cut can land mid-separator; the result must still be valid."""
    name = _collection_name("/" + "a" * 62 + "/" + "b" * 40)
    assert name[-1].isalnum(), name


def test_empty_path_gets_a_fallback_name():
    assert _collection_name("") == "project"


def test_name_never_starts_with_a_digit():
    """Chroma rejects identifiers beginning with a digit."""
    assert not _collection_name("/123/numeric").lstrip("_-")[0].isdigit()


def test_naming_is_deterministic():
    assert _collection_name("/Users/a/p") == _collection_name("/Users/a/p")


def test_different_projects_get_different_collections():
    assert _collection_name("/Users/a/one") != _collection_name("/Users/a/two")


# ── Storage and retrieval ────────────────────────────────────────────────────


def test_storing_nothing_is_a_no_op(isolated_chroma):
    store_chunks("/p", [], [])
    assert not isolated_chroma.exists()


def test_stored_chunk_is_retrievable(isolated_chroma):
    store_chunks("/p", [_chunk(content="def login(): ...")], [_vec(0.1)])
    results = search_chunks("/p", _vec(0.1), n=1)
    assert len(results) == 1
    assert results[0]["content"] == "def login(): ..."


def test_retrieved_chunk_carries_its_location(isolated_chroma):
    """The agent renders these as `file:line` — they must survive the round trip."""
    store_chunks("/p", [_chunk(path="api/auth.py", start=10, end=42)], [_vec(0.2)])
    hit = search_chunks("/p", _vec(0.2), n=1)[0]
    assert hit["file_path"] == "api/auth.py"
    assert hit["start_line"] == 10
    assert hit["end_line"] == 42


def test_results_are_capped_at_n(isolated_chroma):
    chunks = [_chunk(path=f"f{i}.py", start=i, content=f"c{i}") for i in range(10)]
    store_chunks("/p", chunks, [_vec(i / 10) for i in range(10)])
    assert len(search_chunks("/p", _vec(0.0), n=3)) == 3


def test_asking_for_more_than_exists_returns_what_there_is(isolated_chroma):
    store_chunks("/p", [_chunk()], [_vec(0.1)])
    assert len(search_chunks("/p", _vec(0.1), n=50)) == 1


def test_nearest_neighbour_ranks_first(isolated_chroma):
    store_chunks(
        "/p",
        [_chunk(path="a.py", start=1, content="alpha"), _chunk(path="b.py", start=1, content="beta")],
        [[1.0] + [0.0] * (DIM - 1), [0.0] * (DIM - 1) + [1.0]],
    )
    top = search_chunks("/p", [1.0] + [0.0] * (DIM - 1), n=1)[0]
    assert top["content"] == "alpha"


def test_projects_are_isolated_from_each_other(isolated_chroma):
    store_chunks("/one", [_chunk(content="from one")], [_vec(0.1)])
    store_chunks("/two", [_chunk(content="from two")], [_vec(0.1)])
    assert search_chunks("/one", _vec(0.1), n=5)[0]["content"] == "from one"
    assert search_chunks("/two", _vec(0.1), n=5)[0]["content"] == "from two"


def test_reindexing_the_same_chunk_updates_rather_than_duplicates(isolated_chroma):
    """Ids are `file:start_line`, so re-indexing must upsert."""
    store_chunks("/p", [_chunk(content="old body")], [_vec(0.1)])
    store_chunks("/p", [_chunk(content="new body")], [_vec(0.1)])
    results = search_chunks("/p", _vec(0.1), n=10)
    assert len(results) == 1
    assert results[0]["content"] == "new body"


def test_chunks_from_the_same_file_at_different_lines_coexist(isolated_chroma):
    store_chunks(
        "/p",
        [_chunk(start=1, content="top"), _chunk(start=50, content="bottom")],
        [_vec(0.1), _vec(0.9)],
    )
    assert len(search_chunks("/p", _vec(0.1), n=10)) == 2


def test_searching_an_unindexed_project_returns_nothing(isolated_chroma):
    """The grill agent queries before indexing finishes; it must not crash."""
    assert search_chunks("/never/indexed", _vec(0.5), n=5) == []


def test_store_directory_is_created_on_demand(isolated_chroma):
    assert not isolated_chroma.exists()
    store_chunks("/p", [_chunk()], [_vec(0.1)])
    assert isolated_chroma.exists()


@pytest.mark.parametrize("n", [1, 5])
def test_result_dicts_have_a_stable_shape(isolated_chroma, n):
    store_chunks("/p", [_chunk(path=f"f{i}.py", start=i) for i in range(5)],
                 [_vec(i / 10) for i in range(5)])
    for hit in search_chunks("/p", _vec(0.1), n=n):
        assert set(hit) == {"content", "file_path", "start_line", "end_line"}
