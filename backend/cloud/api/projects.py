"""
`/projects`: the server's handle on one indexed folder of a user's.

The runtime keeps the mapping from a local path to a project id. The server
never learns the path, only a name the user can recognise.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from cloud.api.deps import CurrentUser, ServicesDep
from cloud.db.projects import ProjectRecord

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)


class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: datetime

    @classmethod
    def of(cls, record: ProjectRecord) -> ProjectResponse:
        return cls(id=record.id, name=record.name, created_at=record.created_at)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate, user: CurrentUser, services: ServicesDep
) -> ProjectResponse:
    return ProjectResponse.of(await services.projects.create(user.user_id, body.name))


@router.get("")
async def list_projects(user: CurrentUser, services: ServicesDep) -> list[ProjectResponse]:
    return [ProjectResponse.of(record) for record in await services.projects.list_for(user.user_id)]


@router.get("/{project_id}")
async def get_project(
    project_id: uuid.UUID, user: CurrentUser, services: ServicesDep
) -> ProjectResponse:
    record = await services.projects.get(user.user_id, str(project_id))
    if record is None:
        raise _not_found()
    return ProjectResponse.of(record)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID, user: CurrentUser, services: ServicesDep
) -> Response:
    """Delete the project with its chunks and jobs."""
    if not await services.projects.delete(user.user_id, str(project_id)):
        raise _not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _not_found() -> HTTPException:
    # 404 for someone else's project too: whether an id exists is itself not
    # something another user may learn.
    return HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail={"code": "project_not_found", "message": "No such project."},
    )
