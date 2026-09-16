"""
Projects, read and written on a user's behalf.

Every query runs in that user's scope, and none of them filters by owner:
row-level security does that. A query that forgot the filter would still see
only the caller's rows, and a project belonging to someone else is simply not
there.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cloud.db.models import Project
from cloud.db.session import user_scope

_COLUMNS = (Project.id, Project.name, Project.created_at)


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    created_at: datetime


class ProjectRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, user_id: str, name: str) -> ProjectRecord:
        statement = (
            insert(Project)
            .values(id=uuid.uuid4(), owner_id=uuid.UUID(user_id), name=name)
            .returning(*_COLUMNS)
        )
        async with user_scope(self._sessions, user_id) as session:
            row = (await session.execute(statement)).one()
        return _record(row)

    async def list_for(self, user_id: str) -> list[ProjectRecord]:
        statement = select(*_COLUMNS).order_by(Project.created_at, Project.id)
        async with user_scope(self._sessions, user_id) as session:
            rows = (await session.execute(statement)).all()
        return [_record(row) for row in rows]

    async def get(self, user_id: str, project_id: str) -> ProjectRecord | None:
        statement = select(*_COLUMNS).where(Project.id == uuid.UUID(project_id))
        async with user_scope(self._sessions, user_id) as session:
            row = (await session.execute(statement)).one_or_none()
        return None if row is None else _record(row)

    async def delete(self, user_id: str, project_id: str) -> bool:
        """Delete a project with its chunks and jobs. False if the user has no such project."""
        async with user_scope(self._sessions, user_id) as session:
            result = await session.execute(
                delete(Project).where(Project.id == uuid.UUID(project_id))
            )
        return result.rowcount > 0


def _record(row) -> ProjectRecord:
    return ProjectRecord(id=str(row.id), name=row.name, created_at=row.created_at)
