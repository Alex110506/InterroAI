"""
Index jobs as rows: created by the API, advanced by the worker, read back as progress.

The row is the job's source of truth. Its `events` column holds every
`IndexEvent` the worker has produced, in order, so a client that loses its
connection resumes from its position in that list instead of losing progress.
Each append also sends a Postgres NOTIFY on `index_job_events`, carrying the job
id, so a listening API can push the new event at once rather than poll for it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError

from cloud.db.models import ACTIVE_JOB_STATUSES, IndexJob
from cloud.db.session import ScopeFactory
from contracts.indexing import IndexEvent

NOTIFY_CHANNEL = "index_job_events"


@dataclass(frozen=True)
class JobRecord:
    id: str
    project_id: str
    status: str
    upload_ref: str
    events: list[dict]
    error: str | None

    @property
    def settled(self) -> bool:
        return self.status in ("done", "failed")


class ActiveJobExistsError(Exception):
    """The project already has a queued or running job — `job_id` is that job."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Project already has an active index job: {job_id}")
        self.job_id = job_id


class JobRepository:
    def __init__(self, scope: ScopeFactory) -> None:
        self._scope = scope

    async def create(self, project_id: str, upload_ref: str) -> str:
        """
        Queue a job. Raises `ActiveJobExistsError` if the project already has one.

        The check is the partial unique index, not a SELECT first: two requests
        racing each other cannot both pass an index the way they can a read.
        """
        job_id = uuid.uuid4()
        try:
            async with self._scope() as session:
                session.add(
                    IndexJob(id=job_id, project_id=uuid.UUID(project_id), upload_ref=upload_ref)
                )
        except IntegrityError:
            active = await self.active_for(project_id)
            if active is None:
                raise
            raise ActiveJobExistsError(active) from None
        return str(job_id)

    async def active_for(self, project_id: str) -> str | None:
        statement = select(IndexJob.id).where(
            IndexJob.project_id == uuid.UUID(project_id),
            IndexJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        async with self._scope() as session:
            job_id = await session.scalar(statement)
        return None if job_id is None else str(job_id)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._scope() as session:
            job = await session.get(IndexJob, uuid.UUID(job_id))
            if job is None:
                return None
            return JobRecord(
                id=str(job.id),
                project_id=str(job.project_id),
                status=job.status,
                upload_ref=job.upload_ref,
                events=list(job.events),
                error=job.error,
            )

    async def mark_running(self, job_id: str) -> None:
        await self._update(job_id, status="running")

    async def append_event(self, job_id: str, event: IndexEvent) -> int:
        """Append *event* to the job and notify listeners. Returns the new event count."""
        async with self._scope() as session:
            count = await session.scalar(
                text(
                    "UPDATE index_jobs "
                    "SET events = events || jsonb_build_array(CAST(:event AS jsonb)), "
                    "    updated_at = now() "
                    "WHERE id = :id "
                    "RETURNING jsonb_array_length(events)"
                ),
                {"event": event.model_dump_json(exclude_none=True), "id": uuid.UUID(job_id)},
            )
            # Delivered at commit, so a listener never hears of an event it cannot read yet.
            await session.execute(
                text("SELECT pg_notify(:channel, :job_id)"),
                {"channel": NOTIFY_CHANNEL, "job_id": job_id},
            )
        return count or 0

    async def finish(self, job_id: str, *, error: str | None = None) -> None:
        """Settle the job: failed with *error* if one is given, otherwise done."""
        await self._update(job_id, status="failed" if error else "done", error=error)

    async def _update(self, job_id: str, **values) -> None:
        async with self._scope() as session:
            await session.execute(
                update(IndexJob)
                .where(IndexJob.id == uuid.UUID(job_id))
                .values(**values, updated_at=func.now())
            )
