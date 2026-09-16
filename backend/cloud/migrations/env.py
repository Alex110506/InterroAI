"""
Alembic environment for the cloud database.

Migrations run as the database *owner* (INTERROAI_MIGRATIONS_DATABASE_URL),
never as the app role the services connect with: creating extensions, tables
and row-level security policies is exactly what the app role must not be able
to do.
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from cloud.db.models import Base

config = context.config
# Tests drive Alembic in-process and switch this off: fileConfig would
# reconfigure logging underneath pytest's log capture.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    # Set on the config by anything driving Alembic programmatically (tests);
    # read from the environment by everyone else.
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from cloud.settings import MigrationSettings

    return MigrationSettings().migrations_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
