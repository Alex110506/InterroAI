"""
`OpenAIGateway`: the `ModelGateway` that opens straight onto OpenAI.

It uses the platform key its own process configured
(`cloud.llm.openai_client.use_api_key`), and only the Cloud API builds one,
behind sign-in, allowlists and quotas (`cloud/llm/routes.py`). The desktop
runtime always gets `core/remote/gateway.py` instead, which calls that API.
"""
from __future__ import annotations

from typing import Any

import httpx

from cloud.llm.openai_client import chat_completion, chat_stream, get_client
from core.models.gateway import LONG_TIMEOUT


class OpenAIGateway:
    """Direct to OpenAI, with the platform key this process configured."""

    async def chat(self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any):
        return await chat_completion(get_client(timeout), **request)

    async def chat_stream(self, *, timeout: httpx.Timeout = LONG_TIMEOUT, **request: Any):
        return await chat_stream(get_client(timeout), **request)
