"""
The cloud `UploadStore`: Azure Blob Storage (Azurite during development).

One upload is one JSON blob. The runtime writes it straight to storage with a
short-lived SAS URL from the Web API, so megabytes of chunks never pass through
the API; the worker reads it by reference, and deletes it once its job has run.
A lifecycle rule on the container removes anything a failed run leaves behind.
"""
from __future__ import annotations

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient, ContainerClient

from contracts.indexing import ChunkUpload
from core.index.ports import UploadNotFoundError


class BlobUploadStore:
    def __init__(
        self, container: ContainerClient, service: BlobServiceClient | None = None
    ) -> None:
        self._container = container
        self._service = service

    @classmethod
    def from_connection_string(cls, connection_string: str, container_name: str) -> BlobUploadStore:
        service = BlobServiceClient.from_connection_string(connection_string)
        return cls(service.get_container_client(container_name), service)

    async def ensure_container(self) -> None:
        """Create the container if it is missing. Terraform owns it in Azure; this is for dev."""
        try:
            await self._container.create_container()
        except ResourceExistsError:
            pass

    async def put(self, upload_ref: str, upload: ChunkUpload) -> None:
        await self._container.get_blob_client(upload_ref).upload_blob(
            upload.model_dump_json().encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

    async def get(self, upload_ref: str) -> ChunkUpload:
        try:
            downloader = await self._container.get_blob_client(upload_ref).download_blob()
            data = await downloader.readall()
        except ResourceNotFoundError:
            raise UploadNotFoundError(upload_ref) from None
        return ChunkUpload.model_validate_json(data)

    async def delete(self, upload_ref: str) -> None:
        try:
            await self._container.get_blob_client(upload_ref).delete_blob()
        except ResourceNotFoundError:
            pass

    async def close(self) -> None:
        await self._container.close()
        if self._service is not None:
            await self._service.close()
