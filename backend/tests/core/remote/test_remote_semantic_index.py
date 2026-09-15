"""
The runtime's index client, against a scripted stand-in for the Cloud API and
for Blob Storage. The full round trip on the real stack is in
`test_cloud_roundtrip.py`.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from contracts.indexing import (
    Chunk,
    ChunkUpload,
    FileState,
    SearchHit,
    SearchRequest,
    SyncRequest,
)
from core.errors import CloudError, CloudUnavailableError
from core.remote.semantic_index import ProjectRegistry, RemoteSemanticIndex
from core.remote.session import CloudSession
from core.workspace.project_index import embed_project

API = "https://api.example"
PROJECT_PATH = "/Users/someone/code/secret-project"


def _refusal(status: int, code: str, **extra) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"code": code, "message": code, **extra}})


class FakeIndexApi:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.projects: set[str] = set()
        self.max_bytes = 1_000_000
        self.running_job: str | None = None
        self.event_streams: list = []
        self.hits: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        parts = request.url.path.strip("/").split("/")

        if request.url.path == "/projects":
            project_id = f"server-project-{len(self.projects) + 1}"
            self.projects.add(project_id)
            name = json.loads(request.content)["name"]
            return httpx.Response(
                201, json={"id": project_id, "name": name, "created_at": "2026-09-15T12:00:00Z"}
            )

        if parts[0] == "projects":
            project_id, action = parts[1], parts[2]
            if project_id not in self.projects:
                return _refusal(404, "project_not_found")
            if action == "sync":
                return httpx.Response(
                    200, json={"changed": ["a.py"], "removed": [], "unchanged": 0}
                )
            if action == "uploads":
                return httpx.Response(
                    201,
                    json={
                        "upload_ref": f"projects/{project_id}/upload.json",
                        "upload_url": f"https://blob.example/{project_id}/upload.json?sig=x",
                        "expires_at": "2026-09-15T12:15:00Z",
                        "max_bytes": self.max_bytes,
                    },
                )
            if action == "jobs":
                if self.running_job:
                    return _refusal(409, "job_already_running", job_id=self.running_job)
                return httpx.Response(202, json={"job_id": "job-1"})
            if action == "search":
                return httpx.Response(200, json=self.hits)

        if parts[0] == "jobs" and parts[-1] == "events":
            return self.event_streams.pop(0)(request)
        return httpx.Response(404)


class FakeBlob:
    def __init__(self) -> None:
        self.puts: list[httpx.Request] = []
        self.status = 201

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.puts.append(request)
        return httpx.Response(self.status)


@pytest.fixture
def world(tmp_path, fake_keyring):
    api, blob = FakeIndexApi(), FakeBlob()
    session = CloudSession(API, transport=httpx.MockTransport(api))
    session.adopt_tokens(access_token="access", refresh_token="refresh", expires_in=900)
    registry = ProjectRegistry(tmp_path / "cloud_projects.json")
    index = RemoteSemanticIndex(
        session, registry=registry, blob_transport=httpx.MockTransport(blob), reconnect_delay=0
    )
    return SimpleNamespace(api=api, blob=blob, index=index, registry=registry, session=session)


def _sync_request() -> SyncRequest:
    return SyncRequest(project_id=PROJECT_PATH, files=[FileState(file_path="a.py", file_hash="h1")])


def _event_stream(body: bytes, seen: list):
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Last-Event-ID"))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    return respond


def _dropping_stream(first_frame: bytes, seen: list):
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Last-Event-ID"))

        async def body():
            yield first_frame
            raise httpx.ReadError("connection reset")

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    return respond


# ── Projects ─────────────────────────────────────────────────────────────────


async def test_a_folder_becomes_one_server_project_and_stays_it(world):
    await world.index.sync(_sync_request())
    await world.index.sync(_sync_request())

    creations = [r for r in world.api.requests if r.url.path == "/projects"]
    assert len(creations) == 1
    assert world.registry.get(API, PROJECT_PATH) == "server-project-1"


async def test_the_server_is_never_told_the_folders_path(world):
    await world.index.sync(_sync_request())

    for request in world.api.requests:
        assert PROJECT_PATH not in str(request.url)
        assert PROJECT_PATH.encode() not in request.content
    creation = next(r for r in world.api.requests if r.url.path == "/projects")
    assert json.loads(creation.content) == {"name": "secret-project"}


async def test_a_project_deleted_on_the_server_is_made_afresh(world):
    world.registry.set(API, PROJECT_PATH, "deleted-on-the-server")

    result = await world.index.sync(_sync_request())

    assert result.changed == ["a.py"]
    assert world.registry.get(API, PROJECT_PATH) == "server-project-1"


def test_the_registry_keeps_each_apis_projects_apart(tmp_path):
    registry = ProjectRegistry(tmp_path / "projects.json")
    registry.set("https://one.example", "/code/app", "id-1")
    registry.set("https://two.example", "/code/app", "id-2")

    reopened = ProjectRegistry(tmp_path / "projects.json")
    assert reopened.get("https://one.example", "/code/app") == "id-1"
    assert reopened.get("https://two.example", "/code/app") == "id-2"

    reopened.forget("https://one.example", "/code/app")
    assert registry.get("https://one.example", "/code/app") is None


def test_a_corrupt_registry_reads_as_empty(tmp_path):
    path = tmp_path / "projects.json"
    path.write_text("{not json")
    assert ProjectRegistry(path).get(API, PROJECT_PATH) is None


# ── Uploads ──────────────────────────────────────────────────────────────────


async def test_an_upload_goes_straight_to_storage_and_becomes_a_job(world):
    chunk = Chunk(file_path="a.py", start_line=1, end_line=2, file_hash="h1", content="SOURCE-TEXT")

    job_id = await world.index.upload(
        ChunkUpload(project_id=PROJECT_PATH, chunks=[chunk], changed_paths=["a.py"])
    )

    assert job_id == "job-1"
    [put] = world.blob.puts
    assert put.headers["x-ms-blob-type"] == "BlockBlob"
    assert "Authorization" not in put.headers, "the storage URL carries its own signature"
    uploaded = ChunkUpload.model_validate_json(put.content)
    assert uploaded.project_id == "server-project-1"
    assert uploaded.chunks == [chunk]
    assert all(b"SOURCE-TEXT" not in r.content for r in world.api.requests), (
        "source text goes to storage, never through the API"
    )


async def test_an_upload_over_the_limit_is_refused_before_anything_is_sent(world):
    world.api.max_bytes = 10

    with pytest.raises(CloudError) as raised:
        await world.index.upload(ChunkUpload(project_id=PROJECT_PATH, changed_paths=["a.py"]))

    assert raised.value.code == "upload_too_large"
    assert world.blob.puts == []


async def test_a_project_already_being_indexed_is_followed_not_failed(world):
    world.api.running_job = "job-already-running"

    job_id = await world.index.upload(ChunkUpload(project_id=PROJECT_PATH, changed_paths=["a.py"]))

    assert job_id == "job-already-running"


async def test_storage_refusing_the_upload_is_an_error(world):
    world.blob.status = 403

    with pytest.raises(CloudError) as raised:
        await world.index.upload(ChunkUpload(project_id=PROJECT_PATH, changed_paths=["a.py"]))

    assert raised.value.code == "upload_failed"


# ── Job progress ─────────────────────────────────────────────────────────────


async def test_job_events_resume_after_a_dropped_connection(world):
    seen: list = []
    world.api.event_streams = [
        _dropping_stream(b'id: 1\ndata: {"step":"C","status":"start","total":1}\n\n', seen),
        _event_stream(b': keep-alive\n\nid: 2\ndata: {"step":"done","embedded":1}\n\n', seen),
    ]

    events = [event async for event in world.index.job_events("job-1")]

    assert [event.step for event in events] == ["C", "done"]
    assert seen == [None, "1"], "the second connection asks to resume after event 1"


async def test_a_job_stream_that_keeps_dropping_gives_up(world):
    seen: list = []
    world.api.event_streams = [_dropping_stream(b"", seen) for _ in range(10)]

    with pytest.raises(CloudUnavailableError):
        [event async for event in world.index.job_events("job-1")]


# ── Search ───────────────────────────────────────────────────────────────────


async def test_searching_a_folder_never_indexed_asks_the_server_nothing(world):
    assert await world.index.search(SearchRequest(project_id=PROJECT_PATH, query="auth")) == []
    assert world.api.requests == []


async def test_search_returns_the_servers_hits_for_an_indexed_folder(world):
    await world.index.sync(_sync_request())
    world.api.hits = [
        {"file_path": "a.py", "start_line": 1, "end_line": 9, "file_hash": "h1", "score": 0.9}
    ]

    hits = await world.index.search(SearchRequest(project_id=PROJECT_PATH, query="auth"))

    assert hits == [
        SearchHit(file_path="a.py", start_line=1, end_line=9, file_hash="h1", score=0.9)
    ]


# ── What the app sees ────────────────────────────────────────────────────────


async def test_indexing_while_signed_out_ends_in_an_error_event_with_its_code(
    tmp_path, fake_keyring
):
    (tmp_path / "a.py").write_text("print('hi')\n")
    signed_out = RemoteSemanticIndex(
        CloudSession(API, transport=httpx.MockTransport(FakeIndexApi())),
        registry=ProjectRegistry(tmp_path / "cloud_projects.json"),
    )

    events = [event async for event in embed_project(tmp_path, index=signed_out)]

    assert events[-1]["step"] == "error"
    assert events[-1]["code"] == "not_signed_in"
