"""The composition root: which implementation of each port the runtime uses."""
from __future__ import annotations

from core import providers
from core.remote.gateway import RemoteModelGateway
from core.remote.semantic_index import RemoteSemanticIndex


def test_models_and_the_index_both_go_through_the_cloud_api():
    assert isinstance(providers.model_gateway(), RemoteModelGateway)
    assert isinstance(providers.semantic_index(), RemoteSemanticIndex)


def test_every_cloud_client_shares_one_session():
    """One session, so one set of tokens and one refresh at a time for the whole process."""
    assert providers.cloud_session() is providers.cloud_session()


def test_every_caller_gets_its_own_index_handle():
    """A handle's HTTP clients belong to the event loop that opened them."""
    assert providers.semantic_index() is not providers.semantic_index()
