"""
The cloud database schema, as SQLAlchemy models.

The migrations in `cloud/migrations/` create the schema and are its source of
truth. These models are how the API and the worker read and write it, and what
`alembic revision --autogenerate` compares against. Row-level security, the
HNSW index and the role grants exist only in the migrations — SQLAlchemy has no
model for them.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: text-embedding-3-small. The column is fixed-width, so another model is a migration.
EMBEDDING_DIMENSIONS = 1536

JOB_STATUSES = ("queued", "running", "done", "failed")
ACTIVE_JOB_STATUSES = ("queued", "running")


class Base(DeclarativeBase):
    pass


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    #: The identity. Logins can be renamed; the numeric GitHub id cannot.
    github_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    login: Mapped[str] = mapped_column(String(100))
    avatar_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    #: sha256 of the token. The token itself is only ever held by the client.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()


class LoginCode(Base):
    """A one-time code handed to the app's loopback listener, bound to its PKCE challenge."""

    __tablename__ = "login_codes"

    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    code_challenge: Mapped[str] = mapped_column(String(128))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = _now()


class Chunk(Base):
    """A chunk's location, its file's hash and its vector. Never its text."""

    __tablename__ = "chunks"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    file_path: Mapped[str] = mapped_column(Text, primary_key=True)
    start_line: Mapped[int] = mapped_column(Integer, primary_key=True)
    end_line: Mapped[int] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIMENSIONS))


class EmbeddingCacheEntry(Base):
    """
    Vectors by content hash, shared across tenants.

    Deliberately linked to no project and no user: identical text has an
    identical embedding whoever indexed it, and the row holds no text.
    """

    __tablename__ = "embedding_cache"

    model: Mapped[str] = mapped_column(String(100), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = _now()


class IndexJob(Base):
    __tablename__ = "index_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed')", name="index_jobs_status_known"
        ),
        # The database, not the API, guarantees one active job per project: two
        # requests racing each other cannot both win an INSERT.
        Index(
            "one_active_job_per_project",
            "project_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), server_default="queued")
    upload_ref: Mapped[str] = mapped_column(Text)
    #: Every `IndexEvent` so far, in order. A client that reconnects resumes
    #: from its position in this list, so no progress is ever lost to a drop.
    events: Mapped[list[dict]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Usage(Base):
    """Per-user, per-day totals the LLM gateway's quotas are checked against."""

    __tablename__ = "usage"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    requests: Mapped[int] = mapped_column(Integer, server_default="0")
    prompt_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    completion_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
