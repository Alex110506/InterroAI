"""
Incremental Phase 2 indexing: what gets re-embedded, and what gets pruned.

These drive `embed_project` against a real (throwaway) Chroma store, because
the behaviour under test *is* the agreement between the files on disk and the
records in the store — stubbing the store out would test nothing. Only the
embedding call is faked, so no test makes a network request.

The bug these exist to prevent: chunk ids are `file_path:start_line`, so
deleting, renaming or shrinking a file used to leave its old chunks in the
store forever, still answering searches with content the file no longer has.
"""
from __future__ import annotations

import pytest

import core.index.indexer as indexer
import core.workspace.project_index as project_index
from core.index.embeddings import EmbeddedBatch
from core.index.semantic_index import LocalSemanticIndex
from core.index.vector_store import collection_size, search_chunks, stored_manifest


@pytest.fixture
def embedded(monkeypatch):
    """Fake the embedding call and record every text it was asked to embed."""
    calls: list[list[str]] = []

    async def fake_batches(texts, **kwargs):
        calls.append(list(texts))
        yield EmbeddedBatch(
            indices=list(range(len(texts))),
            vectors=[[0.1, 0.2, 0.3, 0.4] for _ in texts],
        )

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    return calls


def _embedded_texts(calls: list[list[str]]) -> list[str]:
    return [text for call in calls for text in call]


async def _passthrough(texts, **kwargs):
    yield EmbeddedBatch(
        indices=list(range(len(texts))),
        vectors=[[0.1, 0.2, 0.3, 0.4] for _ in texts],
    )


async def _index(path, *, force: bool = False, index=None) -> dict:
    """Run a full indexing pass and return its final summary event."""
    summary: dict = {}
    async for event in project_index.embed_project(path, force=force, index=index):
        if event["step"] in {"done", "error"}:
            summary = event
    return summary


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    return root


# ── Reconciliation ───────────────────────────────────────────────────────────


async def test_a_first_run_embeds_everything(project, embedded, isolated_chroma):
    summary = await _index(project)

    assert summary["step"] == "done"
    assert summary["embedded"] > 0
    assert collection_size(str(project)) > 0


async def test_an_unchanged_project_costs_no_api_calls(project, embedded, isolated_chroma):
    """
    The whole reason indexing can run on every startup: re-indexing a project
    nobody touched is a local hashing pass and nothing else.
    """
    await _index(project)
    embedded.clear()

    summary = await _index(project)

    assert embedded == [], "an unchanged project must not be re-embedded"
    assert summary["unchanged"] == 2
    assert summary["embedded"] == 0


async def test_only_the_changed_file_is_re_embedded(project, embedded, isolated_chroma):
    await _index(project)
    embedded.clear()

    (project / "a.py").write_text("def alpha():\n    return 99\n", encoding="utf-8")
    summary = await _index(project)

    texts = _embedded_texts(embedded)
    assert texts, "the changed file must be re-embedded"
    assert any("99" in t for t in texts)
    assert all("beta" not in t for t in texts), "the untouched file must be left alone"
    assert summary["unchanged"] == 1


async def test_a_new_file_is_picked_up(project, embedded, isolated_chroma):
    await _index(project)
    embedded.clear()

    (project / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    await _index(project)

    assert any("gamma" in t for t in _embedded_texts(embedded))


# ── The stale-vector fix ─────────────────────────────────────────────────────


async def test_a_deleted_file_loses_its_vectors(project, embedded, isolated_chroma):
    """
    The reported bug. Upserting alone never removed anything, so a deleted
    file's chunks kept answering searches forever.
    """
    await _index(project)
    assert "b.py" in stored_manifest(str(project))

    (project / "b.py").unlink()
    summary = await _index(project)

    assert "b.py" not in stored_manifest(str(project)), "orphaned chunks must be pruned"
    assert summary["deleted"] > 0


async def test_a_deleted_file_stops_answering_searches(project, embedded, isolated_chroma):
    """The point of pruning: search must not attribute content to a dead file."""
    await _index(project)
    (project / "b.py").unlink()
    await _index(project)

    hits = search_chunks(str(project), [0.1, 0.2, 0.3, 0.4], n=10)
    assert hits, "the surviving file should still be searchable"
    assert all(hit["file_path"] != "b.py" for hit in hits)


async def test_a_renamed_file_leaves_nothing_behind(project, embedded, isolated_chroma):
    await _index(project)

    (project / "b.py").rename(project / "renamed.py")
    await _index(project)

    manifest = stored_manifest(str(project))
    assert "b.py" not in manifest
    assert "renamed.py" in manifest


async def test_a_shrunken_file_loses_the_chunks_past_its_new_end(
    project, embedded, isolated_chroma, monkeypatch
):
    """
    Ids are `file_path:start_line`, so the upsert overwrites the chunks that
    still exist and cannot touch the ones the file no longer reaches.
    """
    # Force several chunks per file so there is something to lose.
    monkeypatch.setattr("core.workspace.chunker._CHUNK_SIZE", 40)
    monkeypatch.setattr("core.workspace.chunker._CHUNK_OVERLAP", 0)

    long_file = project / "a.py"
    long_file.write_text("\n".join(f"def f{i}():\n    return {i}\n" for i in range(40)))
    await _index(project)
    before = len(stored_manifest(str(project))["a.py"].ids)
    assert before > 1, "the fixture needs a multi-chunk file to be meaningful"

    long_file.write_text("def only():\n    return 0\n", encoding="utf-8")
    await _index(project)

    after = len(stored_manifest(str(project))["a.py"].ids)
    assert after < before, "chunks past the new end must be pruned"


async def test_nothing_is_pruned_when_nothing_disappeared(project, embedded, isolated_chroma):
    await _index(project)
    summary = await _index(project)
    assert summary["deleted"] == 0


async def test_deletions_happen_even_with_nothing_to_embed(
    project, embedded, isolated_chroma
):
    """Removing a file is a change, even though it costs no embedding call."""
    await _index(project)
    embedded.clear()

    (project / "a.py").unlink()
    (project / "b.py").unlink()
    summary = await _index(project)

    assert embedded == [], "nothing left to embed"
    assert summary["deleted"] > 0
    assert collection_size(str(project)) == 0


# ── Forced rebuild ───────────────────────────────────────────────────────────


async def test_force_re_embeds_everything(project, embedded, isolated_chroma):
    await _index(project)
    embedded.clear()

    summary = await _index(project, force=True)

    assert _embedded_texts(embedded), "a forced run must not trust the manifest"
    assert summary["unchanged"] == 0


async def test_force_discards_what_the_store_held(project, embedded, isolated_chroma):
    """
    The reason to force is that the store itself is suspect — a corrupt
    collection, or vectors from another embedding model. Reconciling would
    preserve exactly what the user is trying to throw away.
    """
    await _index(project)
    (project / "b.py").unlink()

    await _index(project, force=True)

    assert set(stored_manifest(str(project))) == {"a.py"}


# ── Partial failure ──────────────────────────────────────────────────────────


async def test_batches_already_embedded_survive_a_later_failure(
    project, embedded, isolated_chroma, monkeypatch
):
    """
    The other half of the resilience fix: vectors are stored as each batch
    lands, so a failure late in a run cannot discard what earlier batches
    already paid for.
    """
    async def one_good_batch_then_boom(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[[0.1, 0.2, 0.3, 0.4]])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", one_good_batch_then_boom)
    summary = await _index(project)

    assert summary["step"] == "error"
    assert collection_size(str(project)) == 1, "the first batch must have been kept"


async def test_a_failed_run_is_finished_by_the_next_one(
    project, embedded, isolated_chroma, monkeypatch
):
    """A half-finished index must not read as complete on the next pass."""
    async def only_the_first(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[[0.1, 0.2, 0.3, 0.4]])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", only_the_first)
    await _index(project)

    # Recovered provider: the files that never landed are still pending.
    monkeypatch.setattr(indexer, "embed_batches", _passthrough)
    summary = await _index(project)

    assert summary["step"] == "done"
    assert summary["embedded"] > 0
    assert set(stored_manifest(str(project))) == {"a.py", "b.py"}


# ── Skipped chunks ───────────────────────────────────────────────────────────


async def test_a_skipped_chunk_is_reported_not_hidden(
    project, embedded, isolated_chroma, monkeypatch
):
    """
    A chunk the provider refuses is skipped so the rest of the project still
    indexes — but the summary has to say so, or an incomplete index reads as a
    complete one.
    """
    async def one_fails(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(1, len(texts))),
            vectors=[[0.1, 0.2, 0.3, 0.4] for _ in texts[1:]],
            failures={0: RuntimeError("input too long")},
        )

    monkeypatch.setattr(indexer, "embed_batches", one_fails)
    summary = await _index(project)

    assert summary["step"] == "done"
    assert summary["skipped"] == 1
    assert summary["skipped_files"], "the user needs to know what is not covered"


async def test_a_file_that_embedded_nothing_is_retried_on_the_next_run(
    project, embedded, isolated_chroma, monkeypatch
):
    """
    Nothing was stored, so no hash was either, and reconciliation still sees
    the file as changed — a refusal that turns out to be transient is not
    remembered as a permanent verdict.

    (A *partially* embedded file is the deliberate exception: its surviving
    chunks record the hash, so the run does not re-pay for them every time.
    See the comment beside `skipped` in `_embed_steps`.)
    """
    async def fail_everything(texts, **kwargs):
        yield EmbeddedBatch(
            indices=[],
            vectors=[],
            failures={i: RuntimeError("input too long") for i in range(len(texts))},
        )

    monkeypatch.setattr(indexer, "embed_batches", fail_everything)
    await _index(project)
    assert collection_size(str(project)) == 0

    monkeypatch.setattr(indexer, "embed_batches", _passthrough)
    summary = await _index(project)
    assert summary["embedded"] > 0


# ── Event contract ───────────────────────────────────────────────────────────


async def test_the_reconcile_counts_are_reported(project, embedded, isolated_chroma):
    events = [e async for e in project_index.embed_project(str(project))]
    scan = next(e for e in events if e["step"] == "A" and e.get("status") == "done")

    assert scan["files"] == 2
    assert scan["changed"] == 2
    assert scan["unchanged"] == 0
    assert scan["removed"] == 0


async def test_cache_hits_are_reported(project, isolated_chroma, monkeypatch):
    async def all_cached(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(len(texts))),
            vectors=[[0.1, 0.2, 0.3, 0.4] for _ in texts],
            from_cache=len(texts),
        )

    monkeypatch.setattr(indexer, "embed_batches", all_cached)
    summary = await _index(project)
    assert summary["cached"] == summary["embedded"] > 0


# ── What crosses to the worker ───────────────────────────────────────────────


class SpyIndex:
    """The real local index, recording every sync and upload handed to it."""

    def __init__(self):
        self._inner = LocalSemanticIndex()
        self.syncs: list = []
        self.uploads: list = []

    async def sync(self, request):
        self.syncs.append(request)
        return await self._inner.sync(request)

    async def upload(self, upload):
        self.uploads.append(upload)
        return await self._inner.upload(upload)

    def job_events(self, job_id):
        return self._inner.job_events(job_id)

    async def search(self, request):
        return await self._inner.search(request)


async def test_an_unchanged_project_queues_no_job(project, embedded, isolated_chroma):
    """
    Opening an untouched project must cost a sync and nothing else — in the
    cloud, a job would wake a worker from zero replicas to do no work.
    """
    await _index(project)
    spy = SpyIndex()

    summary = await _index(project, index=spy)

    assert spy.syncs, "the index must still be asked what changed"
    assert spy.uploads == [], "nothing moved, so nothing may be queued"
    assert summary["step"] == "done"


async def test_the_upload_carries_only_the_changed_files(project, embedded, isolated_chroma):
    await _index(project)
    (project / "a.py").write_text("def alpha():\n    return 99\n", encoding="utf-8")
    spy = SpyIndex()

    await _index(project, index=spy)

    [upload] = spy.uploads
    assert upload.changed_paths == ["a.py"]
    assert {chunk.file_path for chunk in upload.chunks} == {"a.py"}
    assert all(chunk.file_hash for chunk in upload.chunks), "each chunk must carry its file hash"


async def test_the_sync_describes_every_file_on_disk(project, embedded, isolated_chroma):
    spy = SpyIndex()
    await _index(project, index=spy)

    [request] = spy.syncs
    assert request.project_id == str(project.resolve())
    assert {f.file_path for f in request.files} == {"a.py", "b.py"}
    assert all(f.file_hash for f in request.files)


async def test_a_file_emptied_since_indexing_loses_its_chunks(project, embedded, isolated_chroma):
    """
    Regression: pruning used to consider only files that produced *new*
    chunks, so a file that was emptied — or grew past the chunker's size cap —
    kept its old vectors forever, and was re-chunked on every run because its
    hash never matched again.
    """
    await _index(project)
    assert "b.py" in stored_manifest(str(project))

    (project / "b.py").write_text("", encoding="utf-8")
    summary = await _index(project)

    assert "b.py" not in stored_manifest(str(project))
    assert summary["deleted"] > 0
