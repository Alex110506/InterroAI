"""
The local SemanticIndex: sync, the queued job, and search.

Everything runs for real — Chroma, the in-memory queue, the worker task —
except the embedding calls, so these exercise the same round trip the cloud
build makes over HTTP and Service Bus.
"""
from __future__ import annotations

import asyncio

import pytest

import core.index.indexer as indexer
import core.index.semantic_index as semantic_index
from contracts.indexing import Chunk, ChunkUpload, FileState, SearchRequest, SyncRequest
from core.index.adapters.chroma import store_chunks, stored_manifest
from core.index.adapters.memory import InMemoryJobQueue, InMemoryUploadStore
from core.index.embeddings import EmbeddedBatch
from core.index.semantic_index import LocalSemanticIndex

VECTOR = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def project(tmp_path) -> str:
    return str(tmp_path / "proj")


@pytest.fixture
def fake_embeddings(monkeypatch):
    async def fake_batches(texts, **kwargs):
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    async def fake_texts(texts, **kwargs):
        return [VECTOR for _ in texts]

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)
    monkeypatch.setattr(semantic_index, "embed_texts", fake_texts)


def _store(project, path, file_hash, start=1):
    store_chunks(
        project,
        [{"file_path": path, "start_line": start, "end_line": start + 4, "file_hash": file_hash}],
        [VECTOR],
    )


def _files(*pairs) -> list[FileState]:
    return [FileState(file_path=path, file_hash=digest) for path, digest in pairs]


def _upload(project) -> ChunkUpload:
    chunk = Chunk(file_path="a.py", start_line=1, end_line=2, file_hash="h1", content="x = 1")
    return ChunkUpload(project_id=project, chunks=[chunk], changed_paths=["a.py"])


async def _events(index, job_id):
    return [event async for event in index.job_events(job_id)]


# ── Sync ─────────────────────────────────────────────────────────────────────


async def test_a_new_project_reports_every_file_as_changed(project, isolated_chroma):
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "h1"), ("b.py", "h2")))
    )
    assert result.changed == ["a.py", "b.py"]
    assert result.removed == []
    assert result.unchanged == 0


async def test_a_file_whose_hash_matches_is_unchanged(project, isolated_chroma):
    _store(project, "a.py", "h1")
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "h1"), ("b.py", "h2")))
    )
    assert result.changed == ["b.py"]
    assert result.unchanged == 1


async def test_a_file_whose_hash_differs_is_changed(project, isolated_chroma):
    _store(project, "a.py", "h1")
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "edited")))
    )
    assert result.changed == ["a.py"]


async def test_a_file_the_client_no_longer_has_is_reported_removed(project, isolated_chroma):
    _store(project, "gone.py", "h1")
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "h1")))
    )
    assert result.removed == ["gone.py"]


async def test_changed_files_keep_the_order_they_were_sent_in(project, isolated_chroma):
    """The client chunks in this order, and chunk order is what batches are cut from."""
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("z.py", "1"), ("a.py", "2"), ("m.py", "3")))
    )
    assert result.changed == ["z.py", "a.py", "m.py"]


async def test_an_unreadable_file_is_never_reported_unchanged(project, isolated_chroma):
    _store(project, "a.py", "")
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "")))
    )
    assert result.changed == ["a.py"]


async def test_force_trusts_nothing_the_index_holds(project, isolated_chroma):
    _store(project, "a.py", "h1")
    _store(project, "gone.py", "h2")
    result = await LocalSemanticIndex().sync(
        SyncRequest(project_id=project, files=_files(("a.py", "h1")), force=True)
    )
    assert result.changed == ["a.py"]
    assert result.removed == [], "the forced job resets the index, so nothing is left to prune"
    assert result.unchanged == 0


# ── Upload and the job ───────────────────────────────────────────────────────


async def test_an_uploaded_job_runs_and_reports_to_the_end(
    project, isolated_chroma, fake_embeddings
):
    index = LocalSemanticIndex()
    events = await _events(index, await index.upload(_upload(project)))

    assert events[-1].step == "done"
    assert "a.py" in stored_manifest(project)


async def test_upload_returns_before_the_job_has_run(project, isolated_chroma, monkeypatch):
    """The client gets a job id at once; the embedding happens on the worker's time."""
    release = asyncio.Event()

    async def slow_batches(texts, **kwargs):
        await release.wait()
        yield EmbeddedBatch(indices=list(range(len(texts))), vectors=[VECTOR for _ in texts])

    monkeypatch.setattr(indexer, "embed_batches", slow_batches)
    index = LocalSemanticIndex()

    job_id = await index.upload(_upload(project))
    assert "a.py" not in stored_manifest(project), "nothing may be stored before the worker runs"

    release.set()
    events = await asyncio.wait_for(_events(index, job_id), timeout=5)
    assert events[-1].step == "done"


async def test_the_queue_carries_a_pointer_and_the_upload_is_dropped_after_use(
    project, isolated_chroma, fake_embeddings
):
    sent = []

    class RecordingQueue(InMemoryJobQueue):
        async def enqueue(self, message):
            sent.append(message)
            await super().enqueue(message)

    uploads = InMemoryUploadStore()
    index = LocalSemanticIndex(queue=RecordingQueue(), uploads=uploads)
    job_id = await index.upload(_upload(project))
    await _events(index, job_id)

    [message] = sent
    assert message.job_id == job_id
    assert message.project_id == project
    assert len(uploads) == 0, "a claim-checked upload is discarded once its job has run"


async def test_a_defect_in_the_worker_still_ends_the_event_stream(
    project, isolated_chroma, monkeypatch
):
    """Otherwise a bug outside the job would leave the client waiting forever."""
    async def broken_run_job(upload, **kwargs):
        raise RuntimeError("worker bug")
        yield  # pragma: no cover

    monkeypatch.setattr(semantic_index, "run_job", broken_run_job)
    queue = InMemoryJobQueue()
    index = LocalSemanticIndex(queue=queue)

    events = await asyncio.wait_for(_events(index, await index.upload(_upload(project))), 5)

    assert events[-1].step == "error"
    assert "worker bug" in events[-1].message
    assert len(queue.dead_letters) == 1, "a defect will not fix itself on redelivery"


async def test_an_unknown_job_reports_an_error(isolated_chroma):
    events = await _events(LocalSemanticIndex(), "no-such-job")
    assert [event.step for event in events] == ["error"]


# ── Search ───────────────────────────────────────────────────────────────────


async def test_search_returns_locations_and_hashes(project, isolated_chroma, fake_embeddings):
    _store(project, "auth.py", "h-auth", start=10)
    hits = await LocalSemanticIndex().search(SearchRequest(project_id=project, query="login"))
    assert [(h.file_path, h.start_line, h.file_hash) for h in hits] == [("auth.py", 10, "h-auth")]


async def test_search_respects_the_requested_count(project, isolated_chroma, fake_embeddings):
    for start in (1, 11, 21, 31, 41):
        _store(project, "a.py", "h", start=start)
    hits = await LocalSemanticIndex().search(SearchRequest(project_id=project, query="x", n=2))
    assert len(hits) == 2


async def test_searching_an_unindexed_project_finds_nothing(
    project, isolated_chroma, fake_embeddings
):
    assert await LocalSemanticIndex().search(SearchRequest(project_id=project, query="x")) == []
