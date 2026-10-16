"""The cloud EmbeddingCache on Postgres. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import pytest
from port_contracts.embedding_cache import EmbeddingCacheContract

from cloud.adapters.pg_cache import PostgresEmbeddingCache
from cloud.db.models import EMBEDDING_DIMENSIONS
from cloud.db.session import service_scope

pytestmark = pytest.mark.integration


class TestPostgresEmbeddingCache(EmbeddingCacheContract):
    @pytest.fixture
    def cache(self, app_sessions):
        return PostgresEmbeddingCache(lambda: service_scope(app_sessions))

    @pytest.fixture
    def dimensions(self):
        return EMBEDDING_DIMENSIONS
