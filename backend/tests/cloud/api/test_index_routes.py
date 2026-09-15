"""
The index routes over HTTP, on the real local stack: Postgres, Azurite, and the
Service Bus emulator where a test needs the real queue. Only query embedding is
faked. Needs the stack: `pytest -m integration`.

Driven through httpx's ASGI transport, so the app runs on the test's own event
loop, which its database connections belong to. That transport hands back a
streamed body only once the stream ends, so the event-stream tests are written
around jobs that settle.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

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
from contracts.indexing import Chunk, ChunkUpload, IndexEvent, IndexJobMessage
from core.index.adapters.memory import InMemoryJobQueue
from core.index.ports import IndexedChunk

pytestmark = pytest.mark.integration

_BLOCK_BLOB = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"}
_github_ids = itertools.count(70_000)


def _axis(position: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[position] = 1.0
    return vector


@pytest.fixture
async def api(app_sessions, blob_uploads):
    signer = TokenSigner("i" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))
    accounts = PostgresAccounts(lambda: anonymous_scope(app_sessions))
    queries: list[str] = []

    async def embed_query(query: str) -> list[float]:
        queries.append(query)
        return _axis(0)

    services = Services(
        signin=None,
        signer=signer,
        accounts=accounts,
        projects=ProjectRepository(app_sessions),
        sessions=app_sessions,
        uploads=blob_uploads,
        queue=InMemoryJobQueue(),
        embed_query=embed_query,
        usage=PostgresUsageMeter(lambda: anonymous_scope(app_sessions)),
        limits=Limits(
            max_upload_bytes=10_000,
            sse_heartbeat_seconds=0.05,
            quota=Quota(requests=1_000, tokens=10**9),
        ),
    )
    transport = httpx.ASGITransport(app=create_app(services))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        async def sign_in(login: str) -> dict[str, str]:
            user = await accounts.upsert_github_user(next(_github_ids), login, None)
            return {"Authorization": f"Bearer {signer.access_token(user.id, user.login)}"}

        async def new_project(headers: dict[str, str]) -> str:
            response = await client.post("/projects", json={"name": "demo"}, headers=headers)
            return response.json()["id"]

        yield SimpleNamespace(
            client=client,
            services=services,
            uploads=blob_uploads,
            queries=queries,
            store=PostgresChunkStore(lambda: service_scope(app_sessions)),
            jobs=JobRepository(lambda: service_scope(app_sessions)),
            sign_in=sign_in,
            new_project=new_project,
        )


def _indexed(path: str, file_hash: str, vector: list[float] | None = None) -> IndexedChunk:
    return IndexedChunk(
        file_path=path, start_line=1, end_line=9, file_hash=file_hash, vector=vector or _axis(5)
    )


def _ref(project: str) -> str:
    return f"projects/{project}/{uuid4().hex}.json"


async def _create_job(api, headers: dict[str, str], project: str, upload_ref: str):
    return await api.client.post(
        f"/projects/{project}/jobs", headers=headers, json={"upload_ref": upload_ref}
    )


def _event_frames(body: str) -> list[tuple[str | None, dict]]:
    frames = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        data = [line[len("data: "):] for line in lines if line.startswith("data: ")]
        if data:
            ids = [line[len("id: "):] for line in lines if line.startswith("id: ")]
            frames.append((ids[0] if ids else None, json.loads(data[0])))
    return frames


# ── Sync ─────────────────────────────────────────────────────────────────────


async def test_sync_reports_what_changed_since_the_index_last_saw_it(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)
    await api.store.apply(
        project,
        upserts=[
            _indexed("kept.py", "h-kept"),
            _indexed("edited.py", "h-old"),
            _indexed("gone.py", "h-gone"),
        ],
        delete=[],
    )

    response = await api.client.post(
        f"/projects/{project}/sync",
        headers=headers,
        json={
            "project_id": project,
            "files": [
                {"file_path": "kept.py", "file_hash": "h-kept"},
                {"file_path": "edited.py", "file_hash": "h-new"},
                {"file_path": "new.py", "file_hash": "h-n"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "changed": ["edited.py", "new.py"],
        "removed": ["gone.py"],
        "unchanged": 1,
    }


async def test_sync_of_someone_elses_project_is_404(api):
    project = await api.new_project(await api.sign_in("alice"))
    bob = await api.sign_in("bob")

    response = await api.client.post(
        f"/projects/{project}/sync", headers=bob, json={"project_id": project, "files": []}
    )

    assert response.status_code == 404


async def test_a_body_naming_another_project_is_refused(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)

    response = await api.client.post(
        f"/projects/{project}/sync", headers=headers, json={"project_id": str(uuid4()), "files": []}
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "project_mismatch"


# ── Uploads and jobs ─────────────────────────────────────────────────────────


async def test_an_upload_granted_and_written_is_queued_as_a_job(api, service_bus_queue):
    api.services.queue = service_bus_queue
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)

    grant = (await api.client.post(f"/projects/{project}/uploads", headers=headers)).json()
    assert grant["upload_ref"].startswith(f"projects/{project}/")
    assert grant["max_bytes"] == 10_000
    upload = ChunkUpload(project_id=project, changed_paths=["a.py"])
    async with httpx.AsyncClient() as http:
        written = await http.put(
            grant["upload_url"], content=upload.model_dump_json(), headers=_BLOCK_BLOB
        )
    assert written.status_code == 201

    created = await api.client.post(
        f"/projects/{project}/jobs", headers=headers, json={"upload_ref": grant["upload_ref"]}
    )

    assert created.status_code == 202
    job_id = created.json()["job_id"]
    delivery = await asyncio.wait_for(service_bus_queue.receive(), 30)
    assert delivery.message == IndexJobMessage(
        job_id=job_id, project_id=project, upload_ref=grant["upload_ref"]
    )
    await delivery.complete()
    status = await api.client.get(f"/jobs/{job_id}", headers=headers)
    assert status.json()["status"] == "queued"


async def test_a_second_job_while_one_is_active_is_409_naming_the_active_one(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)
    first_ref, second_ref = _ref(project), _ref(project)
    for ref in (first_ref, second_ref):
        await api.uploads.put(ref, ChunkUpload(project_id=project))

    first = await _create_job(api, headers, project, first_ref)
    second = await _create_job(api, headers, project, second_ref)

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "job_already_running"
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]


@pytest.mark.parametrize("kind", ["another project's", "not a granted name"])
async def test_a_job_can_only_use_an_upload_granted_for_its_project(api, kind):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)
    ref = _ref(str(uuid4())) if kind == "another project's" else f"projects/{project}/../x.json"

    response = await _create_job(api, headers, project, ref)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_upload_ref"


async def test_a_job_without_its_upload_is_refused(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)

    response = await api.client.post(
        f"/projects/{project}/jobs", headers=headers, json={"upload_ref": _ref(project)}
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "upload_missing"


async def test_an_oversized_upload_is_refused_and_removed(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)
    ref = _ref(project)
    big = Chunk(file_path="big.py", start_line=1, end_line=1, file_hash="h", content="x" * 20_000)
    await api.uploads.put(
        ref, ChunkUpload(project_id=project, chunks=[big], changed_paths=["big.py"])
    )

    response = await _create_job(api, headers, project, ref)

    assert response.status_code == 413
    assert await api.uploads.size(ref) is None


# ── Job progress ─────────────────────────────────────────────────────────────


async def _job_with_events(api, headers, *, settle: bool):
    project = await api.new_project(headers)
    job_id = await api.jobs.create(project, _ref(project))
    await api.jobs.append_event(job_id, IndexEvent(step="C", status="start", total=1))
    if settle:
        await api.jobs.append_event(job_id, IndexEvent(step="done", embedded=1))
        await api.jobs.finish(job_id)
    return job_id


async def test_job_events_stream_the_whole_log_and_end_when_the_job_settles(api):
    headers = await api.sign_in("alice")
    job_id = await _job_with_events(api, headers, settle=True)

    response = await api.client.get(f"/jobs/{job_id}/events", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _event_frames(response.text) == [
        ("1", {"step": "C", "status": "start", "total": 1}),
        ("2", {"step": "done", "embedded": 1}),
    ]


async def test_job_events_resume_after_the_last_event_id(api):
    headers = await api.sign_in("alice")
    job_id = await _job_with_events(api, headers, settle=True)

    response = await api.client.get(
        f"/jobs/{job_id}/events", headers=headers | {"Last-Event-ID": "1"}
    )

    assert _event_frames(response.text) == [("2", {"step": "done", "embedded": 1})]


async def test_job_events_follow_a_running_job_until_it_settles(api):
    headers = await api.sign_in("alice")
    job_id = await _job_with_events(api, headers, settle=False)

    streaming = asyncio.create_task(api.client.get(f"/jobs/{job_id}/events", headers=headers))
    await asyncio.sleep(0.2)
    await api.jobs.append_event(job_id, IndexEvent(step="done", embedded=1))
    await api.jobs.finish(job_id)
    response = await asyncio.wait_for(streaming, 10)

    assert [event["step"] for _, event in _event_frames(response.text)] == ["C", "done"]
    assert ": keep-alive" in response.text


async def test_someone_elses_job_does_not_exist_for_them(api):
    job_id = await _job_with_events(api, await api.sign_in("alice"), settle=True)
    bob = await api.sign_in("bob")

    assert (await api.client.get(f"/jobs/{job_id}", headers=bob)).status_code == 404
    assert (await api.client.get(f"/jobs/{job_id}/events", headers=bob)).status_code == 404


# ── Search ───────────────────────────────────────────────────────────────────


async def test_search_returns_where_the_closest_code_lives(api):
    headers = await api.sign_in("alice")
    project = await api.new_project(headers)
    await api.store.apply(
        project,
        upserts=[_indexed("auth.py", "h-a", _axis(0)), _indexed("billing.py", "h-b", _axis(1))],
        delete=[],
    )

    response = await api.client.post(
        f"/projects/{project}/search",
        headers=headers,
        json={"project_id": project, "query": "where is sign-in handled", "n": 5},
    )

    assert response.status_code == 200
    hits = response.json()
    assert [hit["file_path"] for hit in hits] == ["auth.py", "billing.py"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert "content" not in hits[0]
    assert api.queries == ["where is sign-in handled"]


async def test_search_of_someone_elses_project_is_404_and_embeds_nothing(api):
    project = await api.new_project(await api.sign_in("alice"))
    bob = await api.sign_in("bob")

    response = await api.client.post(
        f"/projects/{project}/search", headers=bob, json={"project_id": project, "query": "x"}
    )

    assert response.status_code == 404
    assert api.queries == []
