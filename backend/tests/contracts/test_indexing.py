"""
The contract models — the shapes both halves of indexing agree on.

What matters most is that a message survives being serialised and read back on
the other side, because the cloud build sends every one of these over HTTP or a
queue.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.indexing import (
    Chunk,
    ChunkUpload,
    FileState,
    IndexEvent,
    IndexJobMessage,
    SearchHit,
    SearchRequest,
    SyncRequest,
    SyncResult,
)

_MESSAGES = [
    SyncRequest(project_id="p", files=[FileState(file_path="a.py", file_hash="h")], force=True),
    SyncResult(changed=["a.py"], removed=["b.py"], unchanged=3),
    ChunkUpload(
        project_id="p",
        chunks=[Chunk(file_path="a.py", start_line=1, end_line=9, file_hash="h", content="x = 1")],
        changed_paths=["a.py"],
        removed_paths=["b.py"],
    ),
    IndexJobMessage(job_id="j", project_id="p", upload_ref="uploads/j"),
    IndexEvent(step="C", status="progress", embedded=4, total=10),
    SearchRequest(project_id="p", query="auth middleware", n=3),
    SearchHit(file_path="a.py", start_line=1, end_line=9, file_hash="h", score=0.5),
]


@pytest.mark.parametrize("message", _MESSAGES, ids=lambda m: type(m).__name__)
def test_every_message_survives_the_wire(message):
    assert type(message).model_validate_json(message.model_dump_json()) == message


def test_a_message_cannot_be_edited_after_it_is_sent():
    message = IndexJobMessage(job_id="j", project_id="p", upload_ref="uploads/j")
    with pytest.raises(ValidationError):
        message.job_id = "someone-elses-job"


def test_a_misspelt_field_fails_where_it_is_sent():
    """Rather than being silently dropped and missed on the other side."""
    with pytest.raises(ValidationError):
        SyncRequest(project_id="p", filez=[])


def test_an_upload_defaults_to_an_empty_incremental_job():
    upload = ChunkUpload(project_id="p")
    assert (upload.chunks, upload.changed_paths, upload.removed_paths) == ([], [], [])
    assert upload.reset is False


def test_a_search_hit_has_nowhere_to_put_source_text():
    assert "content" not in SearchHit.model_fields


def test_a_queue_message_carries_a_pointer_not_the_chunks():
    assert set(IndexJobMessage.model_fields) == {"job_id", "project_id", "upload_ref"}


@pytest.mark.parametrize("n", [0, 11])
def test_the_search_size_is_bounded(n):
    with pytest.raises(ValidationError):
        SearchRequest(project_id="p", query="q", n=n)


def test_an_event_on_the_wire_keeps_zeros_and_omits_the_rest():
    """A zero is information ("nothing deleted"); an absent counter is not."""
    wire = IndexEvent(step="D", status="done", stored=0, deleted=0).to_wire()
    assert wire == {"step": "D", "status": "done", "stored": 0, "deleted": 0}


def test_an_unknown_step_is_rejected():
    with pytest.raises(ValidationError):
        IndexEvent(step="E")
