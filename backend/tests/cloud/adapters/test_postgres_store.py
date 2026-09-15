"""
The cloud ChunkStore on Postgres + pgvector, and the row-level security around it.

Needs the local stack: `pytest -m integration`.
"""
from __future__ import annotations

import pytest
from port_contracts.chunk_store import ChunkStoreContract, unit_vector
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from cloud.adapters.postgres_store import PostgresChunkStore
from cloud.db.models import EMBEDDING_DIMENSIONS
from cloud.db.session import service_scope, user_scope
from core.index.ports import IndexedChunk

pytestmark = pytest.mark.integration


class TestPostgresChunkStore(ChunkStoreContract):
    @pytest.fixture
    def store(self, app_sessions):
        return PostgresChunkStore(lambda: service_scope(app_sessions))

    @pytest.fixture
    def new_project(self, pg_new_project):
        return pg_new_project

    @pytest.fixture
    def dimensions(self):
        return EMBEDDING_DIMENSIONS


def _chunk(path: str, axis: int = 0) -> IndexedChunk:
    return IndexedChunk(path, 1, 10, "h1", unit_vector(EMBEDDING_DIMENSIONS, axis))


@pytest.fixture
async def two_tenants(app_sessions, pg_new_user, pg_new_project):
    """Alice and Bob, each with a project holding one chunk, written by the worker."""
    alice, bob = await pg_new_user("alice"), await pg_new_user("bob")
    alices_project, bobs_project = await pg_new_project(alice), await pg_new_project(bob)

    worker = PostgresChunkStore(lambda: service_scope(app_sessions))
    await worker.apply(alices_project, upserts=[_chunk("alice.py")], delete=[])
    await worker.apply(bobs_project, upserts=[_chunk("bob.py")], delete=[])

    return {"alice": alice, "alices_project": alices_project, "bobs_project": bobs_project}


async def test_a_user_sees_nothing_of_another_users_project(app_sessions, two_tenants):
    """
    The queries never mention an owner — the database refuses on its own, so a
    forgotten WHERE clause in the API returns nothing instead of someone's code.
    """
    as_alice = PostgresChunkStore(lambda: user_scope(app_sessions, two_tenants["alice"]))

    assert await as_alice.manifest(two_tenants["bobs_project"]) == {}
    assert await as_alice.search(
        two_tenants["bobs_project"], unit_vector(EMBEDDING_DIMENSIONS, 0), 10
    ) == []


async def test_a_user_still_sees_their_own_project(app_sessions, two_tenants):
    as_alice = PostgresChunkStore(lambda: user_scope(app_sessions, two_tenants["alice"]))
    assert set(await as_alice.manifest(two_tenants["alices_project"])) == {"alice.py"}


async def test_a_user_cannot_write_into_another_users_project(app_sessions, two_tenants):
    as_alice = PostgresChunkStore(lambda: user_scope(app_sessions, two_tenants["alice"]))

    with pytest.raises(DBAPIError, match="row-level security"):
        await as_alice.apply(two_tenants["bobs_project"], upserts=[_chunk("planted.py")], delete=[])

    worker = PostgresChunkStore(lambda: service_scope(app_sessions))
    assert set(await worker.manifest(two_tenants["bobs_project"])) == {"bob.py"}


async def test_with_no_identity_nothing_is_visible(app_sessions, two_tenants):
    async with app_sessions() as session:
        visible = (await session.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
    assert visible == 0


async def test_the_app_role_can_never_bypass_the_policies(app_sessions):
    """Superusers and BYPASSRLS roles ignore row-level security entirely."""
    async with app_sessions() as session:
        row = (
            await session.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
    assert tuple(row) == (False, False)


async def test_deleting_a_project_deletes_its_chunks(app_sessions, pg_new_project, owner_engine):
    project = await pg_new_project()
    worker = PostgresChunkStore(lambda: service_scope(app_sessions))
    await worker.apply(project, upserts=[_chunk("a.py")], delete=[])

    async with owner_engine.begin() as connection:
        await connection.execute(text("DELETE FROM projects WHERE id = :id"), {"id": project})

    assert await worker.manifest(project) == {}
