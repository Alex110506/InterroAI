"""
Engine construction: the connection pool each process holds.

Every replica of the API and the worker draws on one server's allowance, so the
sizes are deliberately small and set per service in `cloud/settings.py`.
"""
from __future__ import annotations

from cloud.db.session import DEFAULT_MAX_OVERFLOW, DEFAULT_POOL_SIZE, create_engine

URL = "postgresql+asyncpg://interroai_app:secret@localhost:5432/interroai"


def test_the_pool_is_as_big_as_the_caller_asks():
    engine = create_engine(URL, pool_size=3, max_overflow=1)

    assert engine.pool.size() == 3
    assert engine.pool._max_overflow == 1


def test_the_default_pool_leaves_room_for_the_other_replicas():
    """A Postgres B1ms permits 35 user connections, shared by every process."""
    engine = create_engine(URL)

    assert engine.pool.size() == DEFAULT_POOL_SIZE
    assert engine.pool._max_overflow == DEFAULT_MAX_OVERFLOW
    assert DEFAULT_POOL_SIZE + DEFAULT_MAX_OVERFLOW <= 10


def test_a_dead_pooled_connection_is_replaced_rather_than_served():
    assert create_engine(URL).pool._pre_ping is True
