"""
Incremental Phase 2 indexing: what gets re-embedded, and what gets pruned.

These drive `embed_project` against a real store — the in-memory one, held to
the same `ChunkStore` contract pgvector is — because the behaviour under test
*is* the agreement between the files on disk and the records in the store.
Stubbing the store out would test nothing. Only the embedding call is faked, so
no test here makes a network request.

The bug these exist to prevent: chunk ids are `file_path:start_line`, so
deleting or renaming a file — or shrinking one — used to leave its old chunks in
the store forever, still answering searches with content the file no longer has.
"""
from __future__ import annotations

import pytest
from fakes.index import VECTOR, InProcessIndex

import core.index.indexer as indexer
import core.workspace.project_index as project_index
from core.index.embeddings import EmbeddedBatch


def _working_embedder(calls: list[list[str]]):
    """An embedder that succeeds, recording every text it was asked to embed."""

    async def fake_batches(texts, **kwargs):
        calls.append(list(texts))
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    return fake_batches


@pytest.fixture
def index():
    return InProcessIndex()


@pytest.fixture
def embedded(monkeypatch):
    """Fake the embedding call and collect the texts it was handed."""
    calls: list[list[str]] = []
    monkeypatch.setattr(indexer, "embed_batches", _working_embedder(calls))
    return calls


def _texts(calls: list[list[str]]) -> list[str]:
    return [text for call in calls for text in call]


async def _index(path, *, index, force: bool = False) -> dict:
    """Run a full indexing pass and return its final summary event."""
    summary: dict = {}
    async for event in project_index.embed_project(path, force=force, index=index):
        if event["step"] in {"done", "error"}:
            summary = event
    return summary


async def _manifest(index, project) -> dict:
    return await index.store.manifest(str(project))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    return root


# ── Reconciliation ───────────────────────────────────────────────────────────


async def test_a_first_run_embeds_everything(project, embedded, index):
    summary = await _index(project, index=index)

    assert summary["step"] == "done"
    assert summary["embedded"] > 0
    assert len(index.store) > 0


async def test_an_unchanged_project_costs_no_api_calls(project, embedded, index):
    """
    The whole reason indexing can run on every startup: re-indexing a project
    nobody touched is a local hashing pass and nothing else.
    """
    await _index(project, index=index)
    embedded.clear()

    summary = await _index(project, index=index)

    assert embedded == [], "an unchanged project must not be re-embedded"
    assert summary["unchanged"] == 2
    assert summary["embedded"] == 0


async def test_only_the_changed_file_is_re_embedded(project, embedded, index):
    await _index(project, index=index)
    embedded.clear()

    (project / "a.py").write_text("def alpha():\n    return 99\n", encoding="utf-8")
    summary = await _index(project, index=index)

    texts = _texts(embedded)
    assert any("99" in text for text in texts), "the changed file must be re-embedded"
    assert all("beta" not in text for text in texts), "the untouched file must be left alone"
    assert summary["unchanged"] == 1


async def test_a_new_file_is_picked_up(project, embedded, index):
    await _index(project, index=index)
    embedded.clear()

    (project / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    await _index(project, index=index)

    assert any("gamma" in text for text in _texts(embedded))


# ── The stale-vector fix ─────────────────────────────────────────────────────


async def test_a_deleted_file_loses_its_vectors(project, embedded, index):
    """
    The reported bug. Upserting alone never removes anything, so a deleted
    file's chunks kept answering searches forever.
    """
    await _index(project, index=index)
    assert "b.py" in await _manifest(index, project)

    (project / "b.py").unlink()
    summary = await _index(project, index=index)

    assert "b.py" not in await _manifest(index, project), "orphaned chunks must be pruned"
    assert summary["deleted"] > 0


async def test_a_deleted_file_stops_answering_searches(project, embedded, index):
    """The point of pruning: search must not attribute content to a dead file."""
    await _index(project, index=index)
    (project / "b.py").unlink()
    await _index(project, index=index)

    hits = await index.store.search(str(project), VECTOR, 10)

    assert hits, "the surviving file should still be searchable"
    assert all(hit.file_path != "b.py" for hit in hits)


async def test_a_renamed_file_leaves_nothing_behind(project, embedded, index):
    await _index(project, index=index)

    (project / "b.py").rename(project / "renamed.py")
    await _index(project, index=index)

    manifest = await _manifest(index, project)
    assert "b.py" not in manifest
    assert "renamed.py" in manifest


async def test_a_shrunken_file_loses_the_chunks_past_its_new_end(project, embedded, index):
    """
    Ids are `file_path:start_line`, so the upsert overwrites the chunks that
    still exist and cannot touch the ones the file no longer reaches.
    """
    long_file = project / "a.py"
    long_file.write_text(
        "\n".join(f"def f{n}():\n    return {n}\n" for n in range(400)), encoding="utf-8"
    )
    await _index(project, index=index)
    before = len((await _manifest(index, project))["a.py"].ids)
    assert before > 1, "this test is only meaningful with a multi-chunk file"

    long_file.write_text("def only():\n    return 0\n", encoding="utf-8")
    await _index(project, index=index)

    after = len((await _manifest(index, project))["a.py"].ids)
    assert after < before, "chunks past the new end must be pruned"


async def test_nothing_is_pruned_when_nothing_disappeared(project, embedded, index):
    await _index(project, index=index)
    assert (await _index(project, index=index))["deleted"] == 0


async def test_deletions_happen_even_with_nothing_to_embed(project, embedded, index):
    """Removing a file is a change, even though it costs no embedding call."""
    await _index(project, index=index)
    embedded.clear()

    (project / "a.py").unlink()
    (project / "b.py").unlink()
    summary = await _index(project, index=index)

    assert embedded == [], "there is nothing left to embed"
    assert summary["deleted"] > 0
    assert len(index.store) == 0


# ── Forced rebuild ───────────────────────────────────────────────────────────


async def test_force_re_embeds_everything(project, embedded, index):
    await _index(project, index=index)
    embedded.clear()

    summary = await _index(project, force=True, index=index)

    assert _texts(embedded), "a forced run must not trust the manifest"
    assert summary["unchanged"] == 0


async def test_force_discards_what_the_store_held(project, embedded, index):
    """
    The reason to force is that the store itself is suspect — vectors from
    another embedding model, say. Reconciling would preserve exactly what the
    user is trying to throw away.
    """
    await _index(project, index=index)
    (project / "b.py").unlink()

    await _index(project, force=True, index=index)

    assert set(await _manifest(index, project)) == {"a.py"}


# ── Partial failure ──────────────────────────────────────────────────────────


async def test_a_failed_run_changes_nothing_in_the_index(project, embedded, index, monkeypatch):
    """
    The index is written once, after every chunk is embedded, so a provider
    failure part-way through leaves it exactly as it was. What was already paid
    for is banked in the embedding cache instead (see `core/index/indexer.py`).
    """

    async def one_good_batch_then_boom(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[VECTOR])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", one_good_batch_then_boom)
    summary = await _index(project, index=index)

    assert summary["step"] == "error"
    assert len(index.store) == 0, "a failed job must not half-write the index"


async def test_a_failed_run_is_finished_by_the_next_one(project, embedded, index, monkeypatch):
    """A half-finished index must not read as complete on the next pass."""

    async def only_the_first(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[VECTOR])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", only_the_first)
    assert (await _index(project, index=index))["step"] == "error"

    monkeypatch.setattr(indexer, "embed_batches", _working_embedder(embedded))
    summary = await _index(project, index=index)

    assert summary["step"] == "done"
    assert set(await _manifest(index, project)) == {"a.py", "b.py"}
