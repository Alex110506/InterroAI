"""
Pattern 1, end to end, on the local stack.

The runtime's own indexing pipeline (`embed_project`) walks a real folder on
disk and talks to the real Cloud API app through the remote clients. Chunks go
to Azurite through a SAS URL, job rows and vectors to Postgres, and the Embed
Worker runs alongside, taking jobs off the queue. Only OpenAI is faked.

Needs the stack: `pytest -m integration`.
"""
from __future__ import annotations

import asyncio
import itertools
from datetime import timedelta

import httpx
import pytest

import core.index.indexer as indexer
from cloud.adapters.pg_cache import PostgresEmbeddingCache
from cloud.adapters.postgres_store import PostgresChunkStore
from cloud.api.main import create_app
from cloud.api.services import Limits, Services
from cloud.api.tokens import TokenSigner
from cloud.db.accounts import PostgresAccounts
from cloud.db.jobs import JobRepository
from cloud.db.models import EMBEDDING_DIMENSIONS
from cloud.db.projects import ProjectRepository
from cloud.db.session import anonymous_scope, service_scope
from cloud.db.usage import PostgresUsageMeter, Quota
from cloud.worker.runner import JobRunner
from contracts.indexing import SearchRequest
from core.index.adapters.memory import InMemoryJobQueue
from core.index.embeddings import EmbeddedBatch
from core.remote.semantic_index import ProjectRegistry, RemoteSemanticIndex
from core.remote.session import CloudSession
from core.workspace.project_index import embed_project

pytestmark = pytest.mark.integration

CLOUD = "http://cloud.test"
_github_ids = itertools.count(90_000)


def _vector_for(text: str) -> list[float]:
    """A stand-in embedding: text about signing in points one way, everything else another."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0 if "sign_in" in text else 1] = 1.0
    return vector


@pytest.fixture
async def remote_index(app_sessions, blob_uploads, fake_keyring, tmp_path, monkeypatch):
    async def fake_batches(texts, **kwargs):
        yield EmbeddedBatch(
            indices=list(range(len(texts))), vectors=[_vector_for(text) for text in texts]
        )

    monkeypatch.setattr(indexer, "embed_batches", fake_batches)

    def service():
        return service_scope(app_sessions)

    def anonymous():
        return anonymous_scope(app_sessions)

    async def embed_query(query: str) -> list[float]:
        return _vector_for(query)

    queue = InMemoryJobQueue()
    signer = TokenSigner("t" * 32, issuer=CLOUD, access_ttl=timedelta(minutes=15))
    accounts = PostgresAccounts(anonymous)
    services = Services(
        signin=None,
        signer=signer,
        accounts=accounts,
        projects=ProjectRepository(app_sessions),
        sessions=app_sessions,
        uploads=blob_uploads,
        queue=queue,
        embed_query=embed_query,
        usage=PostgresUsageMeter(anonymous),
        limits=Limits(sse_heartbeat_seconds=0.05, quota=Quota(requests=1_000, tokens=10**9)),
    )

    runner = JobRunner(
        queue=queue,
        uploads=blob_uploads,
        store=PostgresChunkStore(service),
        cache=PostgresEmbeddingCache(service),
        jobs=JobRepository(service),
        max_delivery_count=5,
    )
    stop = asyncio.Event()
    worker = asyncio.create_task(runner.run(stop))

    user = await accounts.upsert_github_user(next(_github_ids), "Alex110506", None)
    session = CloudSession(CLOUD, transport=httpx.ASGITransport(app=create_app(services)))
    session.adopt_tokens(
        access_token=signer.access_token(user.id, user.login),
        refresh_token="not-used-here",
        expires_in=900,
    )
    try:
        # No blob transport: uploads really go over the network to Azurite.
        yield RemoteSemanticIndex(
            session, registry=ProjectRegistry(tmp_path / "cloud_projects.json")
        )
    finally:
        stop.set()
        await asyncio.wait_for(worker, 10)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "auth.py").write_text("def sign_in(user):\n    return user.token\n")
    (root / "billing.py").write_text("def charge(invoice):\n    return invoice.total\n")
    return root.resolve()


async def _index(index, root) -> list[dict]:
    return [event async for event in embed_project(root, index=index)]


async def _search(index, root, query: str) -> list[str]:
    hits = await index.search(SearchRequest(project_id=str(root), query=query, n=5))
    return [hit.file_path for hit in hits]


async def test_a_folder_is_indexed_in_the_cloud_and_searched_from_the_runtime(
    remote_index, project
):
    events = await _index(remote_index, project)

    assert events[-1]["step"] == "done", events
    assert events[-1]["embedded"] == 2
    assert (await _search(remote_index, project, "where does sign_in happen"))[0] == "auth.py"


async def test_reindexing_sends_only_what_changed_and_prunes_what_went(remote_index, project):
    await _index(remote_index, project)
    (project / "billing.py").unlink()
    (project / "auth.py").write_text("def sign_in(user):\n    return user.session_token\n")

    events = await _index(remote_index, project)

    synced = next(e for e in events if e["step"] == "A" and e["status"] == "done")
    assert (synced["changed"], synced["removed"]) == (1, 1)
    assert events[-1]["step"] == "done", events
    assert await _search(remote_index, project, "charge an invoice") == ["auth.py"]


async def test_an_untouched_folder_costs_one_sync_and_no_job(remote_index, project):
    await _index(remote_index, project)

    events = await _index(remote_index, project)

    assert events[-1] == {
        "step": "done",
        "embedded": 0,
        "cached": 0,
        "skipped": 0,
        "deleted": 0,
        "unchanged": 2,
    }
