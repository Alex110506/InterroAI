"""
The cloud database migrations.

Rendering them to SQL needs no database, so it runs in every test run and
catches a broken migration before anyone applies it. Applying and reverting
them for real needs the local stack (`pytest -m integration`).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

BACKEND = Path(__file__).resolve().parents[3]


def _config(url: str, **kwargs) -> Config:
    config = Config(str(BACKEND / "cloud" / "alembic.ini"), **kwargs)
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture(scope="module")
def upgrade_sql() -> str:
    config = _config(
        "postgresql+asyncpg://owner:secret@localhost:5432/interroai", output_buffer=io.StringIO()
    )
    command.upgrade(config, "head", sql=True)
    return config.output_buffer.getvalue()


def test_the_vector_extension_is_created_before_the_schema(upgrade_sql):
    # Not before *any* table: Alembic creates its own alembic_version first.
    assert upgrade_sql.index("CREATE EXTENSION IF NOT EXISTS vector") < upgrade_sql.index(
        "CREATE TABLE users"
    )


def test_embeddings_are_fixed_width_vectors(upgrade_sql):
    assert "VECTOR(1536)" in upgrade_sql


def test_chunks_are_searchable_by_cosine_distance_through_hnsw(upgrade_sql):
    assert "USING hnsw (embedding vector_cosine_ops)" in upgrade_sql


def test_the_database_allows_one_active_job_per_project(upgrade_sql):
    assert "one_active_job_per_project" in upgrade_sql
    assert "WHERE status IN ('queued', 'running')" in upgrade_sql


@pytest.mark.parametrize("table", ["projects", "chunks", "index_jobs"])
def test_tenant_tables_enforce_row_level_security(upgrade_sql, table):
    assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in upgrade_sql


def test_the_app_role_gets_data_access_to_application_tables_only(upgrade_sql):
    grant = next(line for line in upgrade_sql.splitlines() if line.startswith("GRANT"))
    assert grant.startswith("GRANT SELECT, INSERT, UPDATE, DELETE ON users")
    assert "alembic_version" not in grant


@pytest.mark.integration
def test_the_schema_tears_down_and_rebuilds_cleanly(migrated_database, database_urls):
    config = _config(database_urls.owner)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
