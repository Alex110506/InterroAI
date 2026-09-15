"""Reading server-sent events, as the Cloud API writes them."""
from __future__ import annotations

from core.remote.sse import ServerSentEvent, read_events


async def _events(*lines: str) -> list[ServerSentEvent]:
    async def stream():
        for line in lines:
            yield line

    return [event async for event in read_events(stream())]


async def test_fields_group_into_events_at_blank_lines():
    events = await _events('id: 1', 'data: {"a":1}', "", "event: error", "data: boom", "")

    assert events == [
        ServerSentEvent(data='{"a":1}', id="1"),
        ServerSentEvent(data="boom", event="error"),
    ]


async def test_comments_are_keep_alives_and_skipped():
    assert await _events(": keep-alive", "", "data: next", "") == [ServerSentEvent(data="next")]


async def test_data_over_several_lines_is_joined():
    [event] = await _events("data: first", "data: second", "")
    assert event.data == "first\nsecond"


async def test_a_value_without_a_space_after_the_colon_is_kept_whole():
    [event] = await _events("data:x", "")
    assert event.data == "x"


async def test_an_event_cut_off_by_the_end_of_the_stream_is_discarded():
    """A connection that dropped mid-frame leaves half a JSON object behind."""
    assert await _events('id: 3', 'data: {"step": "C", "sta') == []
