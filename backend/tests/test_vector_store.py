"""
ChromaDB persistence. The `isolated_chroma` fixture redirects the store into a
tmp directory so no test touches `~/.interroai/chroma`.
"""
from __future__ import annotations

import re

import pytest

from core.vector_store import (
    _collection_name,
    chunk_id,
    collection_size,
    delete_ids,
    reset_collection,
    search_chunks,
    store_chunks,
    stored_manifest,
)

DIM = 8


def _vec(seed: float) -> list[float]:
    return [seed] * DIM


def _chunk(path="main.py", start=1, end=5, content="body", file_hash=None):
    chunk = {"file_path": path, "start_line": start, "end_line": end, "content": content}
    if file_hash is not None:
        chunk["file_hash"] = file_hash
    return chunk


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
        [_chunk(path="a.py", start=1, content="alpha"), _chunk(path="b.py", start=1, content="beta")],  # noqa: E501
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
    """`search_semantic` may query before indexing finishes; it must not crash."""
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


def test_the_file_hash_is_not_leaked_into_search_results(isolated_chroma):
    """It is the store's bookkeeping for re-indexing, not part of a result."""
    store_chunks("/p", [_chunk(file_hash="abc123")], [_vec(0.1)])
    assert "file_hash" not in search_chunks("/p", _vec(0.1), n=1)[0]


# ── The manifest ─────────────────────────────────────────────────────────────


def test_an_unindexed_project_has_an_empty_manifest(isolated_chroma):
    assert stored_manifest("/never/indexed") == {}


def test_the_manifest_reports_the_stored_hash(isolated_chroma):
    store_chunks("/p", [_chunk(file_hash="hash-of-main")], [_vec(0.1)])
    assert stored_manifest("/p")["main.py"].file_hash == "hash-of-main"


def test_the_manifest_groups_every_chunk_of_a_file(isolated_chroma):
    chunks = [_chunk(start=1), _chunk(start=40), _chunk(start=90)]
    store_chunks("/p", chunks, [_vec(0.1), _vec(0.2), _vec(0.3)])

    assert len(stored_manifest("/p")["main.py"].ids) == 3


def test_the_manifest_separates_files(isolated_chroma):
    store_chunks(
        "/p",
        [_chunk(path="a.py", file_hash="ha"), _chunk(path="b.py", file_hash="hb")],
        [_vec(0.1), _vec(0.2)],
    )
    manifest = stored_manifest("/p")

    assert set(manifest) == {"a.py", "b.py"}
    assert manifest["b.py"].file_hash == "hb"


def test_a_chunk_stored_without_a_hash_reports_an_empty_one(isolated_chroma):
    """An index written before hashes existed must reconcile, not crash."""
    store_chunks("/p", [_chunk()], [_vec(0.1)])
    assert stored_manifest("/p")["main.py"].file_hash == ""


def test_the_manifest_ids_match_the_chunk_ids(isolated_chroma):
    chunk = _chunk(path="api/auth.py", start=12)
    store_chunks("/p", [chunk], [_vec(0.1)])
    assert stored_manifest("/p")["api/auth.py"].ids == (chunk_id(chunk),)


# ── Deleting ─────────────────────────────────────────────────────────────────


def test_deleting_ids_removes_them_from_search(isolated_chroma):
    """The stale-vector fix: upserting alone can never remove anything."""
    gone = _chunk(path="deleted.py", content="old content")
    store_chunks("/p", [gone, _chunk(path="kept.py")], [_vec(0.1), _vec(0.2)])

    assert delete_ids("/p", [chunk_id(gone)]) == 1

    paths = {hit["file_path"] for hit in search_chunks("/p", _vec(0.1), n=10)}
    assert paths == {"kept.py"}


def test_deleting_nothing_is_not_an_error(isolated_chroma):
    assert delete_ids("/p", []) == 0


def test_deleting_an_unknown_id_does_not_disturb_the_rest(isolated_chroma):
    store_chunks("/p", [_chunk()], [_vec(0.1)])
    delete_ids("/p", ["main.py:999"])
    assert collection_size("/p") == 1


# ── Resetting ────────────────────────────────────────────────────────────────


def test_resetting_empties_the_collection(isolated_chroma):
    store_chunks("/p", [_chunk()], [_vec(0.1)])
    reset_collection("/p")
    assert collection_size("/p") == 0


def test_resetting_leaves_other_projects_alone(isolated_chroma):
    store_chunks("/one", [_chunk()], [_vec(0.1)])
    store_chunks("/two", [_chunk()], [_vec(0.1)])

    reset_collection("/one")

    assert collection_size("/one") == 0
    assert collection_size("/two") == 1


def test_resetting_an_unindexed_project_is_harmless(isolated_chroma):
    reset_collection("/never/indexed")  # must not raise
