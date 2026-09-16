"""
The worker half of indexing, driven directly with uploads.

Runs against a real store — the in-memory one, held to the same `ChunkStore`
contract pgvector is — because what is under test is the agreement between an
upload and what ends up stored. Only the embedding call is faked. The project
ids point at directories that do not exist: the worker must be able to run
where the repository never was.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.index.embeddings as embeddings
import core.index.indexer as indexer
from contracts.indexing import Chunk, ChunkUpload
from core.errors import MissingAPIKeyError
from core.index.adapters.memory import InMemoryChunkStore, InMemoryEmbeddingCache
from core.index.embeddings import _BATCH_SIZE, EmbeddedBatch
from core.index.ports import IndexedChunk

VECTOR = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def project(tmp_path) -> str:
    return str(tmp_path / "never-created")


@pytest.fixture
def store() -> InMemoryChunkStore:
    return InMemoryChunkStore()


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


async def _store(store, project, *chunks: Chunk) -> None:
    """Put chunks in the store as an earlier job would have."""
    await store.apply(
        project,
        upserts=[
            IndexedChunk(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                file_hash=chunk.file_hash,
                vector=VECTOR,
            )
            for chunk in chunks
        ],
        delete=[],
    )


class SpyStore(InMemoryChunkStore):
    """The real store, counting how many times the index is written."""

    def __init__(self) -> None:
        super().__init__()
        self.applies = 0

    async def apply(self, project_id, **kwargs):
        self.applies += 1
        return await super().apply(project_id, **kwargs)


async def _run(upload: ChunkUpload, *, store, cache=None):
    return [event async for event in indexer.run_job(upload, store=store, cache=cache)]


# ── Storing ──────────────────────────────────────────────────────────────────


async def test_a_job_stores_its_chunks_with_their_hashes(project, embedded, store):
    events = await _run(
        ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]), store=store
    )

    assert events[-1].step == "done"
    assert events[-1].embedded == 1
    assert (await store.manifest(project))["a.py"].file_hash == "h1"


async def test_the_steps_arrive_in_order(project, embedded, store):
    events = await _run(
        ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]), store=store
    )

    assert [(e.step, e.status) for e in events] == [
        ("C", "start"),
        ("C", "progress"),
        ("C", "done"),
        ("D", "start"),
        ("D", "done"),
        ("done", None),
    ]


async def test_a_job_with_nothing_to_embed_makes_no_embedding_call(project, embedded, store):
    await _store(store, project, _chunk(path="gone.py"))
    events = await _run(ChunkUpload(project_id=project, removed_paths=["gone.py"]), store=store)

    assert embedded == []
    assert "C" not in {e.step for e in events}
    assert events[-1].step == "done"


async def test_two_chunks_claiming_one_id_are_stored_once(project, embedded, store, caplog):
    """
    An upload is written by the client, and ids are `file_path:start_line`.
    Postgres refuses a statement that would update one row twice, so the
    collision is settled here — before anything is embedded or stored.
    """
    upload = ChunkUpload(
        project_id=project,
        chunks=[_chunk(start=7, content="first"), _chunk(start=7, content="second")],
        changed_paths=["a.py"],
    )

    with caplog.at_level("WARNING", logger="core.index.indexer"):
        events = await _run(upload, store=store)

    assert embedded == [["second"]], "only the surviving chunk is paid for"
    assert (await store.manifest(project))["a.py"].ids == ("a.py:7",)
    assert events[-1].embedded == 1
    assert "keeping the last of each" in caplog.text


# ── One write, at the end ────────────────────────────────────────────────────


async def test_the_index_is_written_once_however_many_batches(project, monkeypatch):
    async def two_batches(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[VECTOR])
        yield EmbeddedBatch(indices=[1], vectors=[VECTOR])

    monkeypatch.setattr(indexer, "embed_batches", two_batches)
    store = SpyStore()

    await _run(
        ChunkUpload(
            project_id=project, chunks=[_chunk(start=1), _chunk(start=40)], changed_paths=["a.py"]
        ),
        store=store,
    )

    assert store.applies == 1
    assert len((await store.manifest(project))["a.py"].ids) == 2


async def test_a_failed_job_leaves_the_index_exactly_as_it_was(project, store, monkeypatch):
    """Nothing is written until everything is embedded: a search sees before, never half."""
    async def one_batch_then_boom(texts, **kwargs):
        yield EmbeddedBatch(indices=[0], vectors=[VECTOR])
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(indexer, "embed_batches", one_batch_then_boom)
    await _store(store, project, _chunk(path="gone.py"))

    events = await _run(
        ChunkUpload(
            project_id=project,
            chunks=[_chunk(start=1), _chunk(start=40)],
            changed_paths=["a.py"],
            removed_paths=["gone.py"],
        ),
        store=store,
    )

    assert events[-1].step == "error"
    assert "provider exploded" in events[-1].message
    assert set(await store.manifest(project)) == {"gone.py"}, "neither the new chunks nor the prune"


async def test_a_retried_job_pays_only_for_what_never_succeeded(project, store, monkeypatch):
    """Batches finished before the failure are in the cache, so the retry skips them."""
    monkeypatch.setattr(embeddings, "get_client", lambda timeout: object())
    requested: list[int] = []
    fail_on_request = 2

    async def provider(_client, **kwargs):
        requested.append(len(kwargs["input"]))
        if len(requested) == fail_on_request:
            raise RuntimeError("provider exploded")
        return SimpleNamespace(data=[SimpleNamespace(embedding=VECTOR) for _ in kwargs["input"]])

    monkeypatch.setattr(embeddings, "embed_batch", provider)
    cache = InMemoryEmbeddingCache()
    chunks = [_chunk(start=i, content=f"line {i}") for i in range(1, _BATCH_SIZE + 21)]
    upload = ChunkUpload(project_id=project, chunks=chunks, changed_paths=["a.py"])

    first = await _run(upload, store=store, cache=cache)
    assert first[-1].step == "error"

    requested.clear()
    fail_on_request = 0
    second = await _run(upload, store=store, cache=cache)

    assert second[-1].step == "done"
    assert requested == [20], "only the batch that failed may be embedded again"


# ── Pruning ──────────────────────────────────────────────────────────────────


async def test_a_removed_file_is_pruned(project, embedded, store):
    await _store(store, project, _chunk(path="gone.py"))
    events = await _run(ChunkUpload(project_id=project, removed_paths=["gone.py"]), store=store)

    assert "gone.py" not in await store.manifest(project)
    assert events[-1].deleted == 1


async def test_chunks_a_changed_file_no_longer_produces_are_pruned(project, embedded, store):
    await _store(store, project, _chunk(start=1), _chunk(start=40), _chunk(start=90))

    await _run(
        ChunkUpload(
            project_id=project, chunks=[_chunk(start=1, file_hash="h2")], changed_paths=["a.py"]
        ),
        store=store,
    )

    assert (await store.manifest(project))["a.py"].ids == ("a.py:1",)


async def test_a_changed_file_with_no_chunks_left_is_pruned(project, embedded, store):
    """An emptied file sends no chunks, but it is still in `changed_paths`."""
    await _store(store, project, _chunk())
    await _run(ChunkUpload(project_id=project, changed_paths=["a.py"]), store=store)
    assert "a.py" not in await store.manifest(project)


async def test_files_a_job_does_not_mention_are_left_alone(project, embedded, store):
    await _store(store, project, _chunk(path="untouched.py"))
    await _run(
        ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]), store=store
    )
    assert "untouched.py" in await store.manifest(project)


async def test_reset_discards_everything_before_storing(project, embedded, store):
    await _store(store, project, _chunk(path="old.py"))
    await _run(
        ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"], reset=True),
        store=store,
    )
    assert set(await store.manifest(project)) == {"a.py"}


async def test_running_the_same_job_twice_is_harmless(project, embedded, store):
    """Service Bus delivers at least once, so a redelivered job must change nothing more."""
    await _store(store, project, _chunk(path="gone.py"))
    upload = ChunkUpload(
        project_id=project, chunks=[_chunk()], changed_paths=["a.py"], removed_paths=["gone.py"]
    )

    await _run(upload, store=store)
    second = await _run(upload, store=store)

    manifest = await store.manifest(project)
    assert set(manifest) == {"a.py"}
    assert manifest["a.py"].ids == ("a.py:1",)
    assert second[-1].deleted == 0


# ── Failure ──────────────────────────────────────────────────────────────────


async def test_an_expected_failure_is_reported_without_a_traceback(
    project, store, monkeypatch, caplog
):
    async def no_key(texts, **kwargs):
        raise MissingAPIKeyError()
        yield  # pragma: no cover

    monkeypatch.setattr(indexer, "embed_batches", no_key)
    with caplog.at_level("INFO", logger="core.index.indexer"):
        events = await _run(
            ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]), store=store
        )

    assert events[-1].step == "error"
    assert "API key" in events[-1].message
    assert not any(r.exc_info for r in caplog.records)


async def test_an_unexpected_failure_keeps_its_traceback(project, store, monkeypatch, caplog):
    async def boom(texts, **kwargs):
        raise RuntimeError("genuine defect")
        yield  # pragma: no cover

    monkeypatch.setattr(indexer, "embed_batches", boom)
    with caplog.at_level("ERROR", logger="core.index.indexer"):
        events = await _run(
            ChunkUpload(project_id=project, chunks=[_chunk()], changed_paths=["a.py"]), store=store
        )

    assert events[-1].step == "error"
    assert any(r.exc_info for r in caplog.records)


async def test_a_skipped_chunk_is_reported_not_hidden(project, store, monkeypatch):
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
        ),
        store=store,
    )

    assert events[-1].skipped == 1
    assert events[-1].skipped_files == ["huge.py"]


async def test_a_chunk_that_could_not_be_embedded_loses_its_old_version(
    project, store, monkeypatch
):
    """Its old vector describes text the file no longer has."""
    async def first_refused(texts, **kwargs):
        yield EmbeddedBatch(
            indices=[1], vectors=[VECTOR], failures={0: RuntimeError("input too long")}
        )

    monkeypatch.setattr(indexer, "embed_batches", first_refused)
    await _store(
        store, project, _chunk(start=1, file_hash="old"), _chunk(start=40, file_hash="old")
    )

    await _run(
        ChunkUpload(
            project_id=project,
            chunks=[_chunk(start=1, file_hash="new"), _chunk(start=40, file_hash="new")],
            changed_paths=["a.py"],
        ),
        store=store,
    )

    stored = (await store.manifest(project))["a.py"]
    assert stored.ids == ("a.py:40",)
    assert stored.file_hash == "new"
