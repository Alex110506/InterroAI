"""
`SemanticIndex` over the Cloud API: the index lives in pgvector, the files stay here.

  sync        POST /projects/{id}/sync
  upload      POST /projects/{id}/uploads, PUT the chunks to Blob Storage,
              then POST /projects/{id}/jobs
  job_events  GET  /jobs/{id}/events, resumed with Last-Event-ID after a drop
  search      POST /projects/{id}/search

The runtime names a project by its folder on this machine; the server names it by
an id and is never told the path. `ProjectRegistry` keeps the mapping, per Cloud
API, in `~/.interroai/cloud_projects.json`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import httpx

from contracts.indexing import (
    ChunkUpload,
    IndexEvent,
    SearchHit,
    SearchRequest,
    SyncRequest,
    SyncResult,
)
from core.errors import CloudError, CloudUnavailableError
from core.remote.session import CloudSession
from core.remote.sse import read_events

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Longer than the API's keep-alive interval, so a quiet job is not mistaken for a dead one.
_EVENTS_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
_UPLOAD_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_MAX_RECONNECTS = 5


class ProjectRegistry:
    """Which server project each local folder is, per Cloud API."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path.home() / ".interroai" / "cloud_projects.json"

    def get(self, api_url: str, local_path: str) -> str | None:
        return self._load().get(api_url, {}).get(local_path)

    def set(self, api_url: str, local_path: str, project_id: str) -> None:
        data = self._load()
        data.setdefault(api_url, {})[local_path] = project_id
        self._save(data)

    def forget(self, api_url: str, local_path: str) -> None:
        data = self._load()
        if data.get(api_url, {}).pop(local_path, None) is not None:
            self._save(data)

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            return json.loads(self._path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a crash mid-write cannot leave half a file behind.
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(self._path)


class RemoteSemanticIndex:
    def __init__(
        self,
        session: CloudSession,
        *,
        registry: ProjectRegistry | None = None,
        blob_transport: httpx.AsyncBaseTransport | None = None,
        reconnect_delay: float = 1.0,
    ) -> None:
        self._session = session
        self._registry = registry or ProjectRegistry()
        #: Uploads go straight to Blob Storage, not through the API.
        self._blob_transport = blob_transport
        self._reconnect_delay = reconnect_delay

    # ── Sync ─────────────────────────────────────────────────────────────────

    async def sync(self, request: SyncRequest) -> SyncResult:
        async def call(project_id: str) -> SyncResult:
            body = request.model_copy(update={"project_id": project_id}).model_dump(mode="json")
            response = await self._session.request(
                "POST", f"/projects/{project_id}/sync", json=body
            )
            return SyncResult.model_validate(response.json())

        return await self._with_project(request.project_id, call, create=True)

    # ── Upload and job progress ──────────────────────────────────────────────

    async def upload(self, upload: ChunkUpload) -> str:
        async def call(project_id: str) -> str:
            grant = (await self._session.request("POST", f"/projects/{project_id}/uploads")).json()
            body = upload.model_copy(update={"project_id": project_id}).model_dump_json()
            payload = body.encode("utf-8")
            if len(payload) > grant["max_bytes"]:
                raise CloudError(
                    f"These changes come to {len(payload):,} bytes, over the "
                    f"{grant['max_bytes']:,}-byte upload limit.",
                    code="upload_too_large",
                )
            await self._put_blob(grant["upload_url"], payload)

            try:
                response = await self._session.request(
                    "POST",
                    f"/projects/{project_id}/jobs",
                    json={"upload_ref": grant["upload_ref"]},
                )
            except CloudError as exc:
                if exc.code == "job_already_running" and exc.details.get("job_id"):
                    # The same project opened twice. Follow the job already
                    # running; the next sync picks up whatever it did not cover.
                    logger.info(
                        "Project %s is already being indexed; following that job", project_id
                    )
                    return str(exc.details["job_id"])
                raise
            return str(response.json()["job_id"])

        return await self._with_project(upload.project_id, call, create=True)

    async def job_events(self, job_id: str) -> AsyncIterator[IndexEvent]:
        last_event_id: str | None = None
        reconnects = 0
        while True:
            headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
            try:
                async with self._session.stream(
                    "GET", f"/jobs/{job_id}/events", headers=headers, timeout=_EVENTS_TIMEOUT
                ) as response:
                    async for event in read_events(response.aiter_lines()):
                        if event.id is not None:
                            last_event_id = event.id
                        reconnects = 0
                        yield IndexEvent.model_validate_json(event.data)
                # The API closes the stream once the job has settled.
                return
            except (httpx.TransportError, CloudUnavailableError) as exc:
                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    raise CloudUnavailableError(
                        "Lost the connection to the indexing job and could not get it back."
                    ) from exc
                logger.info(
                    "Job %s event stream dropped (%s); resuming after event %s",
                    job_id,
                    exc,
                    last_event_id,
                )
                await asyncio.sleep(self._reconnect_delay * reconnects)

    # ── Search ───────────────────────────────────────────────────────────────

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        async def call(project_id: str) -> list[SearchHit]:
            body = request.model_copy(update={"project_id": project_id}).model_dump(mode="json")
            response = await self._session.request(
                "POST", f"/projects/{project_id}/search", json=body
            )
            return [SearchHit.model_validate(hit) for hit in response.json()]

        # A folder never indexed has nothing to find, and is not worth a project.
        return await self._with_project(request.project_id, call, create=False) or []

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _with_project(
        self, local_path: str, call: Callable[[str], Awaitable[T]], *, create: bool
    ) -> T | None:
        """
        Run *call* with the server's id for *local_path*.

        A project the server no longer has (deleted there, or lost with its
        database) is forgotten here. With *create*, a new one is made and the
        call retried, which re-indexes the folder from nothing; without, the
        answer is None.
        """
        api_url = self._session.api_url
        project_id = self._registry.get(api_url, local_path)
        if project_id is None:
            if not create:
                return None
            project_id = await self._create_project(local_path)

        try:
            return await call(project_id)
        except CloudError as exc:
            if exc.code != "project_not_found":
                raise

        self._registry.forget(api_url, local_path)
        if not create:
            return None
        return await call(await self._create_project(local_path))

    async def _create_project(self, local_path: str) -> str:
        # The folder's name, so the user can recognise it; never its full path.
        name = (Path(local_path).name or "project")[:200]
        response = await self._session.request("POST", "/projects", json={"name": name})
        project_id = str(response.json()["id"])
        self._registry.set(self._session.api_url, local_path, project_id)
        return project_id

    async def _put_blob(self, upload_url: str, payload: bytes) -> None:
        async with httpx.AsyncClient(
            timeout=_UPLOAD_TIMEOUT, transport=self._blob_transport
        ) as blob:
            try:
                response = await blob.put(
                    upload_url,
                    content=payload,
                    headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"},
                )
            except httpx.TransportError as exc:
                raise CloudUnavailableError("The upload to InterroAI's storage failed.") from exc
        if not response.is_success:
            # Azure names the reason in a header: ContainerNotFound,
            # AuthenticationFailed (an expired URL), and so on.
            reason = response.headers.get("x-ms-error-code")
            status = f"{response.status_code} {reason}" if reason else str(response.status_code)
            raise CloudError(f"Storage refused the upload ({status}).", code="upload_failed")
