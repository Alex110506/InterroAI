"""The composition root: which implementation of each port each build uses."""
from __future__ import annotations

from core import providers
from core.index.semantic_index import LocalSemanticIndex
from core.models.gateway import OpenAIGateway
from core.remote.gateway import RemoteModelGateway
from core.remote.semantic_index import RemoteSemanticIndex


def test_the_cloud_build_goes_through_the_cloud_api(cloud_mode):
    assert isinstance(providers.model_gateway(), RemoteModelGateway)
    assert isinstance(providers.semantic_index(), RemoteSemanticIndex)


def test_every_cloud_client_shares_one_session(cloud_mode):
    """One session, so one set of tokens and one refresh at a time for the whole process."""
    assert providers.cloud_session() is providers.cloud_session()
    assert providers.cloud_session().api_url == "https://api.example"


def test_the_local_build_calls_openai_directly():
    assert isinstance(providers.model_gateway(), OpenAIGateway)


def test_the_local_build_indexes_in_process():
    assert isinstance(providers.semantic_index(), LocalSemanticIndex)


def test_every_caller_gets_its_own_index_handle():
    """A local handle's queue and worker tasks belong to one event loop."""
    assert providers.semantic_index() is not providers.semantic_index()
