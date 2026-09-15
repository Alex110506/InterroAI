"""
The composition root: which implementation of each port this process uses.

Everything that needs a `ModelGateway` or a `SemanticIndex` takes one as an
argument and, when given none, asks here. That keeps the choice in exactly one
place. `INTERROAI_MODE` makes it: `local` talks to OpenAI directly and indexes
into Chroma on disk; `cloud` goes through the Cloud API for both. Tests hand in
their own fakes explicitly instead of patching module globals.
"""
from __future__ import annotations

from functools import lru_cache

from core.index.semantic_index import LocalSemanticIndex, SemanticIndex
from core.models.gateway import ModelGateway, OpenAIGateway
from core.remote.gateway import RemoteModelGateway
from core.remote.semantic_index import RemoteSemanticIndex
from core.remote.session import CloudSession
from core.settings import get_runtime_settings


def model_gateway() -> ModelGateway:
    if get_runtime_settings().mode == "cloud":
        return RemoteModelGateway(cloud_session())
    return OpenAIGateway()


def semantic_index() -> SemanticIndex:
    """
    A fresh index handle.

    Fresh rather than shared because a local handle owns a queue and worker
    tasks bound to the event loop that first uses them.
    """
    if get_runtime_settings().mode == "cloud":
        return RemoteSemanticIndex(cloud_session())
    return LocalSemanticIndex()


@lru_cache(maxsize=1)
def cloud_session() -> CloudSession:
    """
    The process's one session with the Cloud API.

    Shared, because every cloud client must use the same tokens: two sessions
    refreshing independently would spend the same rotating refresh token twice,
    which the API treats as theft.
    """
    return CloudSession(get_runtime_settings().api_url)
