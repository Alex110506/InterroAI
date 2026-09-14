"""
The composition root: which implementation of each port this process uses.

Everything that needs a `ModelGateway` or a `SemanticIndex` takes one as an
argument and, when given none, asks here. That keeps the choice in exactly one
place — the cloud build changes these functions (by configuration) and nothing
else — while tests hand in their own fakes explicitly instead of patching
module globals.
"""
from __future__ import annotations

from core.index.semantic_index import LocalSemanticIndex, SemanticIndex
from core.models.gateway import ModelGateway, OpenAIGateway


def model_gateway() -> ModelGateway:
    return OpenAIGateway()


def semantic_index() -> SemanticIndex:
    """
    A fresh index handle.

    Fresh rather than shared because a local handle owns a queue and worker
    tasks bound to the event loop that first uses them.
    """
    return LocalSemanticIndex()
