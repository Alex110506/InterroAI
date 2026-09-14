"""The composition root: which implementation of each port the local build uses."""
from __future__ import annotations

from core import providers
from core.index.semantic_index import LocalSemanticIndex
from core.models.gateway import OpenAIGateway


def test_the_local_build_calls_openai_directly():
    assert isinstance(providers.model_gateway(), OpenAIGateway)


def test_the_local_build_indexes_in_process():
    assert isinstance(providers.semantic_index(), LocalSemanticIndex)


def test_every_caller_gets_its_own_index_handle():
    """A local handle's queue and worker tasks belong to one event loop."""
    assert providers.semantic_index() is not providers.semantic_index()
