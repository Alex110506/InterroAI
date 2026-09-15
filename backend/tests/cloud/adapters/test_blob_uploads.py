"""The cloud UploadStore on Azurite. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from port_contracts.upload_store import UploadStoreContract

from cloud.adapters.blob_uploads import BlobUploadStore
from contracts.indexing import ChunkUpload
from core.index.ports import UnusableUploadError, UploadNotFoundError

pytestmark = pytest.mark.integration

_BLOCK_BLOB = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"}


class TestBlobUploadStore(UploadStoreContract):
    @pytest.fixture
    def uploads(self, blob_uploads):
        return blob_uploads


def _ref() -> str:
    return f"tests/{uuid4().hex}.json"


def _in(minutes: float) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


async def test_an_upload_is_stored_as_plain_json(blob_uploads):
    """The runtime writes these itself with a SAS URL, so the format is the contract."""
    upload_ref = _ref()
    await blob_uploads.put(upload_ref, ChunkUpload(project_id="p", changed_paths=["a.py"]))

    blob = blob_uploads._container.get_blob_client(upload_ref)
    raw = await (await blob.download_blob()).readall()

    assert json.loads(raw)["changed_paths"] == ["a.py"]


async def test_ensuring_the_container_creates_it_once_and_then_does_nothing(
    blob_uploads, stack_settings
):
    """The API does this at startup, so a fresh stack accepts uploads and a restart is harmless."""
    store = BlobUploadStore.from_connection_string(
        stack_settings.blob_connection_string, f"interroai-test-{uuid4().hex[:12]}"
    )
    try:
        await store.ensure_container()
        await store.ensure_container()

        upload_ref = _ref()
        await store.put(upload_ref, ChunkUpload(project_id="p", changed_paths=["a.py"]))
        assert (await store.get(upload_ref)).changed_paths == ["a.py"]
    finally:
        await store._container.delete_container()
        await store.close()


# ── Upload URLs ──────────────────────────────────────────────────────────────


async def test_an_upload_url_lets_its_holder_write_the_blob_but_not_read_it(blob_uploads):
    upload_ref = _ref()
    url = blob_uploads.upload_url(upload_ref, expires_at=_in(5))
    body = ChunkUpload(project_id="p", changed_paths=["a.py"]).model_dump_json()

    async with httpx.AsyncClient() as http:
        written = await http.put(url, content=body, headers=_BLOCK_BLOB)
        read_back = await http.get(url)

    assert written.status_code == 201
    assert read_back.status_code == 403
    assert (await blob_uploads.get(upload_ref)).changed_paths == ["a.py"]


async def test_an_upload_url_covers_only_its_own_blob(blob_uploads):
    mine = _ref()
    url = blob_uploads.upload_url(mine, expires_at=_in(5))

    async with httpx.AsyncClient() as http:
        response = await http.put(url.replace(mine, _ref()), content=b"{}", headers=_BLOCK_BLOB)

    assert response.status_code == 403


async def test_an_expired_upload_url_is_refused(blob_uploads):
    upload_ref = _ref()
    url = blob_uploads.upload_url(upload_ref, expires_at=_in(-1))

    async with httpx.AsyncClient() as http:
        response = await http.put(url, content=b"{}", headers=_BLOCK_BLOB)

    assert response.status_code == 403
    assert await blob_uploads.size(upload_ref) is None


# ── Untrusted content ────────────────────────────────────────────────────────


async def test_size_is_in_bytes_and_none_when_nothing_is_there(blob_uploads):
    upload_ref = _ref()
    upload = ChunkUpload(project_id="p", changed_paths=["a.py"])
    await blob_uploads.put(upload_ref, upload)

    assert await blob_uploads.size(upload_ref) == len(upload.model_dump_json().encode("utf-8"))
    assert await blob_uploads.size(_ref()) is None


async def test_an_upload_over_the_limit_is_unusable(blob_uploads, stack_settings):
    upload_ref = _ref()
    await blob_uploads.put(upload_ref, ChunkUpload(project_id="p", changed_paths=["a.py"]))
    limited = BlobUploadStore.from_connection_string(
        stack_settings.blob_connection_string, "interroai-test-uploads", max_bytes=10
    )
    try:
        with pytest.raises(UnusableUploadError, match="limit") as raised:
            await limited.get(upload_ref)
    finally:
        await limited.close()

    assert not isinstance(raised.value, UploadNotFoundError)


async def test_a_malformed_upload_is_unusable(blob_uploads):
    upload_ref = _ref()
    await blob_uploads._container.get_blob_client(upload_ref).upload_blob(b'{"chunks": 1}')

    with pytest.raises(UnusableUploadError) as raised:
        await blob_uploads.get(upload_ref)

    assert not isinstance(raised.value, UploadNotFoundError)
