"""
A reader for server-sent events, as the Cloud API sends them.

Only the parts the API uses: `id:`, `event:` and `data:` fields, and `:`
comments, which are keep-alives and are skipped. An event is complete at a blank
line; one still incomplete when the stream ends is discarded, as the SSE spec
requires, because a connection that dropped mid-frame leaves half a JSON object.
"""
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSentEvent:
    data: str
    event: str = "message"
    id: str | None = None


async def read_events(lines: AsyncIterable[str]) -> AsyncIterator[ServerSentEvent]:
    """Group lines, without their line endings, into events."""
    data: list[str] = []
    event = "message"
    event_id: str | None = None

    async for line in lines:
        if not line:
            if data:
                yield ServerSentEvent(data="\n".join(data), event=event, id=event_id)
            data, event, event_id = [], "message", None
            continue
        if line.startswith(":"):
            continue

        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
        elif name == "id":
            event_id = value
