"""The cloud UploadStore on Azurite. Needs the local stack: `pytest -m integration`."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest
from port_contracts.upload_store import UploadStoreContract

from contracts.indexing import ChunkUpload

pytestmark = pytest.mark.integration


class TestBlobUploadStore(UploadStoreContract):
    @pytest.fixture
    def uploads(self, blob_uploads):
        return blob_uploads


async def test_an_upload_is_stored_as_plain_json(blob_uploads):
    """The runtime writes these itself with a SAS URL, so the format is the contract."""
    upload_ref = f"tests/{uuid4().hex}.json"
    await blob_uploads.put(upload_ref, ChunkUpload(project_id="p", changed_paths=["a.py"]))

    blob = blob_uploads._container.get_blob_client(upload_ref)
    raw = await (await blob.download_blob()).readall()

    assert json.loads(raw)["changed_paths"] == ["a.py"]
