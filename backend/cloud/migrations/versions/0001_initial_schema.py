"""Initial schema: users and sign-in, projects, chunks, the embedding cache, jobs, usage.

Also the parts SQLAlchemy models cannot express: the pgvector extension, the
HNSW index, row-level security, and the app role's grants.

Revision ID: 0001
Revises:
Create Date: 2026-09-15
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "interroai_app"
DIMENSIONS = 1536

_TABLES = (
    "users",
    "refresh_tokens",
    "login_codes",
    "projects",
    "chunks",
    "embedding_cache",
    "index_jobs",
    "usage",
)


def _timestamp(name: str, **kwargs) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), **kwargs)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("github_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("login", sa.String(100), nullable=False),
        sa.Column("avatar_url", sa.Text()),
        _timestamp("created_at", nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        _timestamp("expires_at", nullable=False),
        _timestamp("revoked_at"),
        _timestamp("created_at", nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "login_codes",
        sa.Column("code_hash", sa.String(64), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code_challenge", sa.String(128), nullable=False),
        _timestamp("expires_at", nullable=False),
        _timestamp("used_at"),
    )

    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        _timestamp("created_at", nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("file_path", sa.Text(), primary_key=True),
        sa.Column("start_line", sa.Integer(), primary_key=True),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("embedding", VECTOR(DIMENSIONS), nullable=False),
    )
    # Approximate nearest-neighbour search by cosine distance. Filtering by
    # project happens after the index scan, which is why searches turn on
    # iterative scanning (see cloud/adapters/postgres_store.py).
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "embedding_cache",
        sa.Column("model", sa.String(100), primary_key=True),
        sa.Column("content_hash", sa.String(64), primary_key=True),
        sa.Column("embedding", VECTOR(DIMENSIONS), nullable=False),
        _timestamp("created_at", nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "index_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("upload_ref", sa.Text(), nullable=False),
        sa.Column(
            "events",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("error", sa.Text()),
        _timestamp("created_at", nullable=False, server_default=sa.func.now()),
        _timestamp("updated_at", nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed')", name="index_jobs_status_known"
        ),
    )
    op.create_index(
        "one_active_job_per_project",
        "index_jobs",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )

    op.create_table(
        "usage",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
    )

    _enable_row_level_security()
    _grant_the_app_role()


def _enable_row_level_security() -> None:
    # Who is asking, as set per transaction by cloud/db/session.py. NULLIF
    # because a transaction-local setting reads back as '' — not NULL — on a
    # pooled connection that carried one before, and ''::uuid is an error.
    op.execute(
        """
        CREATE FUNCTION app_user_id() RETURNS uuid LANGUAGE sql STABLE AS
        $$ SELECT NULLIF(current_setting('app.user_id', true), '')::uuid $$
        """
    )
    op.execute(
        """
        CREATE FUNCTION app_is_worker() RETURNS boolean LANGUAGE sql STABLE AS
        $$ SELECT coalesce(current_setting('app.service', true), '') = 'worker' $$
        """
    )

    owns_project = (
        "app_is_worker() OR EXISTS ("
        "SELECT 1 FROM projects p WHERE p.id = {table}.project_id AND p.owner_id = app_user_id())"
    )

    op.execute("ALTER TABLE projects ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY projects_by_owner ON projects "
        "USING (app_is_worker() OR owner_id = app_user_id()) "
        "WITH CHECK (app_is_worker() OR owner_id = app_user_id())"
    )

    for table in ("chunks", "index_jobs"):
        condition = owns_project.format(table=table)
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_by_project_owner ON {table} "
            f"USING ({condition}) WITH CHECK ({condition})"
        )


def _grant_the_app_role() -> None:
    # The role normally exists already (infra/local/postgres/init/01-roles.sh
    # locally, Terraform in Azure). Creating it without LOGIN here only keeps
    # the migration applicable to a bare database; it grants no way in.
    op.execute(
        f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN;
            END IF;
        END $$
        """
    )
    # Data access on the application's tables only — not alembic_version, and
    # no DDL: the app role can never change the schema or the policies.
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(_TABLES)} TO {APP_ROLE}")


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.drop_table(table)
    op.execute("DROP FUNCTION IF EXISTS app_is_worker()")
    op.execute("DROP FUNCTION IF EXISTS app_user_id()")
    # The extension stays: other databases on the server may rely on it.
