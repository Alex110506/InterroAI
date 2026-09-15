"""
The cloud `UploadStore`: Azure Blob Storage (Azurite during development).

One upload is one JSON blob. The runtime writes it straight to storage through
a short-lived SAS URL from the Web API, so megabytes of chunks never pass
through the API. The worker reads it by reference and deletes it once its job
has run; a lifecycle rule on the container removes anything a failed run leaves
behind.

The blob is written by the user, so the worker treats it as untrusted input: it
is size-checked before it is read and validated as it is parsed, and both
failures are final (`UnusableUploadError`). Retrying would only fail the same
way again.
"""
from __future__ import annotations

from datetime import datetime

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from pydantic import ValidationError

from contracts.indexing import ChunkUpload
from core.index.ports import UnusableUploadError, UploadNotFoundError


class BlobUploadStore:
    def __init__(
        self,
        container: ContainerClient,
        service: BlobServiceClient | None = None,
        *,
        account_name: str | None = None,
        account_key: str | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self._container = container
        self._service = service
        #: Sign upload URLs. Without them the store still reads and writes,
        #: but cannot hand out URLs.
        self._account_name = account_name
        self._account_key = account_key
        self._max_bytes = max_bytes

    @classmethod
    def from_connection_string(
        cls, connection_string: str, container_name: str, *, max_bytes: int | None = None
    ) -> BlobUploadStore:
        service = BlobServiceClient.from_connection_string(connection_string)
        return cls(
            service.get_container_client(container_name),
            service,
            account_name=_connection_setting(connection_string, "AccountName"),
            account_key=_connection_setting(connection_string, "AccountKey"),
            max_bytes=max_bytes,
        )

    async def ensure_container(self) -> None:
        """Create the container if it is missing. Terraform owns it in Azure; this is for dev."""
        try:
            await self._container.create_container()
        except ResourceExistsError:
            pass

    def upload_url(self, upload_ref: str, *, expires_at: datetime) -> str:
        """
        A URL that lets its holder write this one blob until *expires_at*.

        Create and write only. The holder can neither read the blob back nor
        touch any other blob, so a leaked URL exposes nothing already uploaded.
        """
        if not (self._account_name and self._account_key):
            raise RuntimeError("This store has no account key, so it cannot sign upload URLs.")
        sas = generate_blob_sas(
            account_name=self._account_name,
            container_name=self._container.container_name,
            blob_name=upload_ref,
            account_key=self._account_key,
            permission=BlobSasPermissions(create=True, write=True),
            expiry=expires_at,
        )
        return f"{self._container.get_blob_client(upload_ref).url}?{sas}"

    async def size(self, upload_ref: str) -> int | None:
        """The upload's size in bytes, or None if nothing is stored under *upload_ref*."""
        try:
            properties = await self._container.get_blob_client(upload_ref).get_blob_properties()
        except ResourceNotFoundError:
            return None
        return properties.size

    async def put(self, upload_ref: str, upload: ChunkUpload) -> None:
        await self._container.get_blob_client(upload_ref).upload_blob(
            upload.model_dump_json().encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

    async def get(self, upload_ref: str) -> ChunkUpload:
        blob = self._container.get_blob_client(upload_ref)
        try:
            if self._max_bytes is not None:
                size = (await blob.get_blob_properties()).size
                if size > self._max_bytes:
                    raise UnusableUploadError(
                        f"The upload is {size} bytes; the limit is {self._max_bytes}."
                    )
            data = await (await blob.download_blob()).readall()
        except ResourceNotFoundError:
            raise UploadNotFoundError(upload_ref) from None

        try:
            return ChunkUpload.model_validate_json(data)
        except ValidationError as exc:
            raise UnusableUploadError(
                f"The upload is not a valid chunk upload ({exc.error_count()} problems)."
            ) from None

    async def delete(self, upload_ref: str) -> None:
        try:
            await self._container.get_blob_client(upload_ref).delete_blob()
        except ResourceNotFoundError:
            pass

    async def close(self) -> None:
        await self._container.close()
        if self._service is not None:
            await self._service.close()


def _connection_setting(connection_string: str, name: str) -> str | None:
    for part in connection_string.split(";"):
        # partition, not split: an account key is base64 and ends in "=".
        key, _, value = part.partition("=")
        if key.strip().lower() == name.lower():
            return value.strip()
    return None
