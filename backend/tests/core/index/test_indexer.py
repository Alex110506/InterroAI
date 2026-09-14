"""
The worker half of indexing, driven directly with uploads.

Runs against a real throwaway Chroma store — what is under test is the
agreement between an upload and what ends up stored — with only the embedding
call faked. The project ids point at directories that do not exist: the worker
must be able to run where the repository never was.
"""
from __future__ import annotations

import pytest

import core.index.indexer as indexer
from contracts.indexing import Chunk, ChunkUpload
from core.errors import MissingAPIKeyError
from core.index.embeddings import EmbeddedBatch
from core.index.vector_store import collection_size, store_chunks, stored_manifest

VECTOR = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def project(tmp_path) -> str:
    return str(tmp_path / "never-created")


@pytest.fixture
def embedded(monkeypatch):
    """Fake the embedding call and record every text it was asked to embed."""
    calls: list[list[str]] = []

    async def fake_batches(texts, **kwargs):
        calls.append(list(texts))
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    return calls


def _chunk(path="a.py", start=1, file_hash="h1", content="x = 1") -> Chunk:
    return Chunk(
        file_path=path, start_line=start, end_line=start + 4, file_hash=file_hash, content=content
    )


def _store(project, *chunks: Chunk) -> None:
    """Put chunks in the store as an earlier job would have."""
    store_chunks(project, [c.model_dump() for c in chunks], [VECTOR for _ in chunks])


async def _run(upload: ChunkUpload):
    return [event async for event in indexer.run_job(upload)]


# ── Storing ──────────────────────────────────────────────────────────────────


async def test_a_job_stores_its_chunks_with_their_hashes(project, embedded, isolated_chroma):
    events = await _run(ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]))

    assert events[-1].step == "done"
    assert events[-1].embedded == 1
    assert stored_manifest(project)["a.py"].file_hash == "h1"


async def test_the_steps_arrive_in_order(project, embedded, isolated_chroma):
    events = await _run(ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]))

    assert [(e.step, e.status) for e in events] == [
        ("C", "start"),
        ("C", "progress"),
        ("C", "done"),
        ("D", "start"),
        ("D", "done"),
        ("done", None),
    ]


async def test_a_job_with_nothing_to_embed_makes_no_embedding_call(
    project, embedded, isolated_chroma
):
    _store(project, _chunk(path="gone.py"))
    events = await _run(ChunkUpload(project_id=project, removed_paths=["gone.py"]))

    assert embedded == []
    assert "C" not in {e.step for e in events}
    assert events[-1].step == "done"


# ── Pruning ──────────────────────────────────────────────────────────────────


async def test_a_removed_file_is_pruned(project, embedded, isolated_chroma):
    _store(project, _chunk(path="gone.py"))
    events = await _run(ChunkUpload(project_id=project, removed_paths=["gone.py"]))

    assert "gone.py" not in stored_manifest(project)
    assert events[-1].deleted == 1


async def test_chunks_a_changed_file_no_longer_produces_are_pruned(
    project, embedded, isolated_chroma
):
    _store(project, _chunk(start=1), _chunk(start=40), _chunk(start=90))

    await _run(
        ChunkUpload(project_id=project, chunks=[_chunk(start=1, file_hash="h2")], changed_paths=["a.py"])  # noqa: E501
    )

    assert stored_manifest(project)["a.py"].ids == ("a.py:1",)


async def test_a_changed_file_with_no_chunks_left_is_pruned(project, embedded, isolated_chroma):
    """An emptied file sends no chunks, but it is still in `changed_paths`."""
    _store(project, _chunk())
    await _run(ChunkUpload(project_id=project, changed_paths=["a.py"]))
    assert "a.py" not in stored_manifest(project)


async def test_files_a_job_does_not_mention_are_left_alone(project, embedded, isolated_chroma):
    _store(project, _chunk(path="untouched.py"))
    await _run(ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]))
    assert "untouched.py" in stored_manifest(project)


async def test_reset_discards_everything_before_storing(project, embedded, isolated_chroma):
    _store(project, _chunk(path="old.py"))
    await _run(
        ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"], reset=True)
    )
    assert set(stored_manifest(project)) == {"a.py"}


async def test_running_the_same_job_twice_is_harmless(project, embedded, isolated_chroma):
    """Service Bus delivers at least once, so a redelivered job must change nothing more."""
    _store(project, _chunk(path="gone.py"))
    upload = ChunkUpload(
        project_id=project, chunks=[_chunk()], changed_paths=["a.py"], removed_paths=["gone.py"]
    )

    await _run(upload)
    second = await _run(upload)

    assert set(stored_manifest(project)) == {"a.py"}
    assert stored_manifest(project)["a.py"].ids == ("a.py:1",)
    assert second[-1].deleted == 0


# ── Failure ──────────────────────────────────────────────────────────────────


async def test_batches_already_stored_survive_a_later_failure(
    project, isolated_chroma, monkeypatch
):
    async def one_batch_then_boom(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[VECTOR])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", one_batch_then_boom)
    events = await _run(
        ChunkUpload(
            project_id=project,
            chunks=[_chunk(start=1), _chunk(start=40)],
            changed_paths=["a.py"],
        )
    )

    assert events[-1].step == "error"
    assert "provider exploded" in events[-1].message
    assert collection_size(project) == 1


async def test_a_failed_job_prunes_nothing(project, isolated_chroma, monkeypatch):
    """Pruning comes after storing, so a failure leaves old vectors, not a hole."""
    async def boom(texts, **kwargs):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(indexer, "embed_batches", boom)
    _store(project, _chunk(path="gone.py"))

    await _run(
        ChunkUpload(
            project_id=project, chunks=[_chunk()], changed_paths=["a.py"], removed_paths=["gone.py"]
        )
    )

    assert "gone.py" in stored_manifest(project)


async def test_an_expected_failure_is_reported_without_a_traceback(
    project, isolated_chroma, monkeypatch, caplog
):
    async def no_key(texts, **kwargs):
        raise MissingAPIKeyError()
        yield  # pragma: no cover

    monkeypatch.setattr(indexer, "embed_batches", no_key)
    with caplog.at_level("INFO", logger="core.index.indexer"):
        events = await _run(
            ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"])
        )

    assert events[-1].step == "error"
    assert "API key" in events[-1].message
    assert not any(r.exc_info for r in caplog.records)


async def test_an_unexpected_failure_keeps_its_traceback(
    project, isolated_chroma, monkeypatch, caplog
):
    async def boom(texts, **kwargs):
        raise RuntimeError("genuine defect")
        yield  # pragma: no cover

    monkeypatch.setattr(indexer, "embed_batches", boom)
    with caplog.at_level("ERROR", logger="core.index.indexer"):
        events = await _run(
            ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"])
        )

    assert events[-1].step == "error"
    assert any(r.exc_info for r in caplog.records)


async def test_a_skipped_chunk_is_reported_not_hidden(project, isolated_chroma, monkeypatch):
    async def first_refused(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(1, len(texts))),
            vectors=[VECTOR for _ in texts[1:]],
            failures={0: RuntimeError("input too long")},
        )

    monkeypatch.setattr(indexer, "embed_batches", first_refused)
    events = await _run(
        ChunkUpload(
            project_id=project,
            chunks=[_chunk(path="huge.py"), _chunk(path="fine.py")],
            changed_paths=["huge.py", "fine.py"],
        )
    )

    assert events[-1].skipped == 1
    assert events[-1].skipped_files == ["huge.py"]
