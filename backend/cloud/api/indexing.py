"""
The index's HTTP face: the arrows between the runtime and the indexing service.

  POST /projects/{id}/sync       which files changed since the index last saw them
  POST /projects/{id}/uploads    a write-only URL for this run's chunks
  POST /projects/{id}/jobs       queue the uploaded chunks for the Embed Worker
  GET  /jobs/{id}                a job's status and its events so far
  GET  /jobs/{id}/events         the job's progress, as server-sent events
  POST /projects/{id}/search     where matching code lives

Reads and writes run in the caller's scope, so row-level security, not these
handlers, is what keeps one user out of another's index. The handlers still look
the project up first, so the answer is a 404 rather than an empty result.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from cloud.adapters.postgres_store import PostgresChunkStore
from cloud.api.deps import CurrentUser, ServicesDep, api_error, charge_request, rate_limit
from cloud.api.job_events import job_event_stream
from cloud.api.llm_gateway import upstream_errors
from cloud.api.services import Services
from cloud.api.tokens import AccessClaims
from cloud.db.jobs import ActiveJobExistsError, JobRepository
from cloud.db.session import ScopeFactory, user_scope
from contracts.indexing import (
    IndexJobMessage,
    SearchHit,
    SearchRequest,
    SyncRequest,
    SyncResult,
)
from core.index.manifest_diff import diff_manifest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["index"])

SyncUser = Annotated[AccessClaims, Depends(rate_limit("sync"))]
UploadUser = Annotated[AccessClaims, Depends(rate_limit("uploads"))]
SearchUser = Annotated[AccessClaims, Depends(rate_limit("search"))]

#: The shape `grant_upload` hands out: the project's prefix, then a random name.
_UPLOAD_REF = re.compile(r"projects/[0-9a-f-]{36}/[0-9a-f]{32}\.json")
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class UploadGrant(BaseModel):
    upload_ref: str
    #: PUT the `ChunkUpload` JSON here, with the header `x-ms-blob-type: BlockBlob`.
    upload_url: str
    expires_at: datetime
    max_bytes: int


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_ref: str


class JobAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    id: str
    project_id: str
    status: str
    events: list[dict]
    error: str | None


# ── Sync ─────────────────────────────────────────────────────────────────────


@router.post("/projects/{project_id}/sync")
async def sync(
    project_id: uuid.UUID, body: SyncRequest, user: SyncUser, services: ServicesDep
) -> SyncResult:
    project = await _owned_project(services, user, project_id, named_in_body=body.project_id)
    indexed = {} if body.force else await PostgresChunkStore(_scope(services, user)).manifest(
        project
    )
    return diff_manifest(
        body.files,
        {path: stored.file_hash for path, stored in indexed.items()},
        force=body.force,
    )


# ── Upload and job ───────────────────────────────────────────────────────────


@router.post("/projects/{project_id}/uploads", status_code=status.HTTP_201_CREATED)
async def grant_upload(
    project_id: uuid.UUID, user: UploadUser, services: ServicesDep
) -> UploadGrant:
    project = await _owned_project(services, user, project_id)
    upload_ref = f"projects/{project}/{uuid.uuid4().hex}.json"
    expires_at = datetime.now(UTC) + services.limits.upload_url_ttl
    return UploadGrant(
        upload_ref=upload_ref,
        upload_url=services.uploads.upload_url(upload_ref, expires_at=expires_at),
        expires_at=expires_at,
        max_bytes=services.limits.max_upload_bytes,
    )


@router.post("/projects/{project_id}/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    project_id: uuid.UUID, body: JobCreate, user: UploadUser, services: ServicesDep
) -> JobAccepted:
    project = await _owned_project(services, user, project_id)
    upload_ref = body.upload_ref
    # Only an upload granted for this project: never another project's, whose
    # chunks the worker would otherwise index here.
    if not (_UPLOAD_REF.fullmatch(upload_ref) and upload_ref.startswith(f"projects/{project}/")):
        raise api_error(400, "invalid_upload_ref", "That upload was not granted for this project.")

    size = await services.uploads.size(upload_ref)
    if size is None:
        raise api_error(400, "upload_missing", "Nothing has been uploaded to that reference.")
    if size > services.limits.max_upload_bytes:
        await services.uploads.delete(upload_ref)
        raise api_error(
            413,
            "upload_too_large",
            f"The upload is {size} bytes; the limit is {services.limits.max_upload_bytes}.",
        )

    jobs = JobRepository(_scope(services, user))
    try:
        job_id = await jobs.create(project, upload_ref)
    except ActiveJobExistsError as exc:
        raise api_error(
            409,
            "job_already_running",
            "This project is already being indexed.",
            job_id=exc.job_id,
        ) from None

    try:
        await services.queue.enqueue(
            IndexJobMessage(job_id=job_id, project_id=project, upload_ref=upload_ref)
        )
    except Exception as exc:
        # A row no worker will ever pick up would block the project's next job.
        logger.exception("Could not queue index job %s", job_id)
        await jobs.finish(job_id, error="The job could not be queued.")
        raise api_error(
            503, "queue_unavailable", "Indexing is unavailable right now. Try again shortly."
        ) from exc
    return JobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, user: CurrentUser, services: ServicesDep) -> JobStatus:
    job = await JobRepository(_scope(services, user)).get(str(job_id))
    if job is None:
        raise _job_not_found()
    return JobStatus(
        id=job.id, project_id=job.project_id, status=job.status, events=job.events, error=job.error
    )


@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: uuid.UUID,
    user: CurrentUser,
    services: ServicesDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """
    The job's events as they happen, ending once it has settled.

    Authorised when it opens: a stream outlives the access token that opened
    it, which is fine for something that ends when its job does.
    """
    jobs = JobRepository(_scope(services, user))
    key = str(job_id)
    if await jobs.get(key) is None:
        raise _job_not_found()
    after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0

    async def frames():
        async with services.notifications.subscribe(key) as signal:
            async for frame in job_event_stream(
                lambda: jobs.get(key),
                signal,
                after=after,
                heartbeat_seconds=services.limits.sse_heartbeat_seconds,
            ):
                yield frame

    return StreamingResponse(frames(), media_type="text/event-stream", headers=_STREAM_HEADERS)


# ── Search ───────────────────────────────────────────────────────────────────


@router.post("/projects/{project_id}/search")
async def search(
    project_id: uuid.UUID, body: SearchRequest, user: SearchUser, services: ServicesDep
) -> list[SearchHit]:
    project = await _owned_project(services, user, project_id, named_in_body=body.project_id)
    if not body.query.strip():
        raise api_error(400, "invalid_request", "The search query is empty.")

    # Embedding the query spends the platform key, so it is metered like chat.
    await charge_request(services, user)
    with upstream_errors():
        vector = await services.embed_query(body.query)
    return await PostgresChunkStore(_scope(services, user)).search(project, vector, body.n)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scope(services: Services, user: AccessClaims) -> ScopeFactory:
    return lambda: user_scope(services.sessions, user.user_id)


async def _owned_project(
    services: Services,
    user: AccessClaims,
    project_id: uuid.UUID,
    *,
    named_in_body: str | None = None,
) -> str:
    project = str(project_id)
    if named_in_body is not None and named_in_body != project:
        raise api_error(
            400, "project_mismatch", "The request body names a different project than the URL."
        )
    if await services.projects.get(user.user_id, project) is None:
        raise api_error(404, "project_not_found", "No such project.")
    return project


def _job_not_found():
    return api_error(404, "job_not_found", "No such index job.")
