"""
One job, end to end across the real local stack: a blob in Azurite, a message on
the Service Bus emulator, a job row and chunks in Postgres. Only the embedding
call is faked. Needs the stack: `pytest -m integration`.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import core.index.indexer as indexer
from cloud.adapters.pg_cache import PostgresEmbeddingCache
from cloud.adapters.postgres_store import PostgresChunkStore
from cloud.db.jobs import JobRepository
from cloud.db.models import EMBEDDING_DIMENSIONS
from cloud.db.session import service_scope
from cloud.worker.runner import JobRunner
from contracts.indexing import Chunk, ChunkUpload, IndexJobMessage
from core.index.embeddings import EmbeddedBatch
from core.index.ports import UploadNotFoundError

pytestmark = pytest.mark.integration


async def test_a_queued_job_is_indexed_end_to_end(
    app_sessions, pg_new_project, blob_uploads, service_bus_queue, monkeypatch
):
    async def fake_batches(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(len(texts))),
            vectors=[[0.01] * EMBEDDING_DIMENSIONS for _ in texts],
        )

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)

    def scope():
        return service_scope(app_sessions)

    project = await pg_new_project()
    jobs = JobRepository(scope)
    upload_ref = f"tests/{uuid4().hex}.json"
    chunk = Chunk(file_path="a.py", start_line=1, end_line=3, file_hash="h1", content="x = 1")
    await blob_uploads.put(
        upload_ref, ChunkUpload(project_id=project, chunks=[chunk], changed_paths=["a.py"])
    )
    job_id = await jobs.create(project, upload_ref)
    await service_bus_queue.enqueue(
        IndexJobMessage(job_id=job_id, project_id=project, upload_ref=upload_ref)
    )

    runner = JobRunner(
        queue=service_bus_queue,
        uploads=blob_uploads,
        store=PostgresChunkStore(scope),
        cache=PostgresEmbeddingCache(scope),
        jobs=jobs,
        max_delivery_count=5,
    )
    await runner.handle(await asyncio.wait_for(service_bus_queue.receive(), 30))

    job = await jobs.get(job_id)
    assert job.status == "done"
    assert job.events[-1]["step"] == "done"
    assert "a.py" in await PostgresChunkStore(scope).manifest(project)
    with pytest.raises(UploadNotFoundError):
        await blob_uploads.get(upload_ref)
