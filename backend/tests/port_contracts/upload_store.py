"""
What every `UploadStore` must do identically — a dict locally, Blob Storage in the cloud.

Subclass `UploadStoreContract` and provide `uploads`.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from contracts.indexing import Chunk, ChunkUpload
from core.index.ports import UploadNotFoundError


def _ref() -> str:
    return f"contract-tests/{uuid4().hex}.json"


def _upload() -> ChunkUpload:
    return ChunkUpload(
        project_id="project-1",
        chunks=[
            Chunk(
                file_path="src/app.py",
                start_line=1,
                end_line=12,
                file_hash="h1",
                content="def main():\n    return 'héllo'\n",
            )
        ],
        changed_paths=["src/app.py"],
        removed_paths=["gone.py"],
        reset=False,
    )


class UploadStoreContract:
    async def test_an_upload_comes_back_exactly_as_it_was_put(self, uploads):
        ref, upload = _ref(), _upload()
        await uploads.put(ref, upload)
        assert await uploads.get(ref) == upload

    async def test_reading_a_missing_upload_raises(self, uploads):
        with pytest.raises(UploadNotFoundError):
            await uploads.get(_ref())

    async def test_a_deleted_upload_is_gone(self, uploads):
        ref = _ref()
        await uploads.put(ref, _upload())
        await uploads.delete(ref)
        with pytest.raises(UploadNotFoundError):
            await uploads.get(ref)

    async def test_deleting_a_missing_upload_is_harmless(self, uploads):
        await uploads.delete(_ref())
