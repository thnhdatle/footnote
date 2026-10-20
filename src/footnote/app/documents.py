# SPDX-License-Identifier: Apache-2.0
"""Documents use case: upload with synchronous checks, read, patch, delete, text,
originals and chunks.

An upload is checked before anything is stored: size (`file_too_large`), media type
sniffed from content (`unsupported_media_type`), encrypted PDFs, page and archive limits
(the `UploadInspector` port, implemented by the parsers adapter), then the tenant quotas
`max_upload_bytes_per_day` and `max_documents` (`quota_exceeded`). The repository stores
the original and enqueues the `ingest` job in one transaction, or reports `unchanged`
when the file, title and chunking settings equal those of the current version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import orjson

from footnote.domain.documents import (
    ChunkDetail,
    ChunkId,
    CollectionId,
    DocumentId,
    DocumentRecord,
    Element,
    NewUpload,
    UploadResult,
    VersionRecord,
)
from footnote.domain.errors import ConflictError, TooLargeError, ValidationError
from footnote.domain.ports import (
    ChunkReader,
    CollectionRepository,
    DocumentRepository,
    IndexRepository,
    TenantReader,
    UploadInspector,
)
from footnote.domain.tenancy import TenantId

METADATA_MAX_BYTES = 16 * 1024
TITLE_MAX_CHARS = 512
EXTERNAL_ID_MAX_CHARS = 512
DOCUMENT_NOT_READY = "document_not_ready"


@dataclass(frozen=True, slots=True)
class DocumentConfig:
    """`ingest.max_file_bytes` (the other upload limits are fixed)."""

    max_file_bytes: int = 50 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DocumentDeps:
    """Ports of the documents use case; `tenants` (admin pool) enables quotas."""

    collections: CollectionRepository
    documents: DocumentRepository
    index: IndexRepository
    chunks: ChunkReader
    inspector: UploadInspector
    tenants: TenantReader | None = None


@dataclass(frozen=True, slots=True)
class UploadRequest:
    """One multipart upload; None fields keep the existing document's values.

    `ingest_delay` is evaluation-only (not an API field): see `NewUpload`.
    """

    filename: str
    data: bytes
    external_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None
    ingest_delay: timedelta = timedelta(0)


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """A document with its current chunk count and versions (single-document reads)."""

    record: DocumentRecord
    chunk_count: int
    versions: tuple[VersionRecord, ...]


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """A JSON object of at most 16 KiB serialised."""
    if not isinstance(metadata, dict):
        raise ValidationError("metadata", "must be a JSON object")
    try:
        size = len(orjson.dumps(metadata))
    except TypeError as exc:
        raise ValidationError("metadata", "must be JSON-serialisable") from exc
    if size > METADATA_MAX_BYTES:
        raise ValidationError("metadata", f"must be at most {METADATA_MAX_BYTES} bytes as JSON")
    return metadata


def validate_title(title: str) -> str:
    """At most 512 characters."""
    if len(title) > TITLE_MAX_CHARS:
        raise ValidationError("title", f"must be at most {TITLE_MAX_CHARS} characters")
    return title


def _validate_external_id(external_id: str) -> str:
    if not 1 <= len(external_id) <= EXTERNAL_ID_MAX_CHARS or not external_id.isprintable():
        raise ValidationError(
            "external_id", f"must be 1..{EXTERNAL_ID_MAX_CHARS} printable characters"
        )
    return external_id


class DocumentService:
    """Documents of a tenant's collections."""

    def __init__(self, deps: DocumentDeps, config: DocumentConfig | None = None) -> None:
        self._d = deps
        self._cfg = config or DocumentConfig()

    @property
    def max_file_bytes(self) -> int:
        """Upload size limit (the API stops reading the body beyond it)."""
        return self._cfg.max_file_bytes

    async def upload(
        self, tenant: TenantId, collection_id: CollectionId, request: UploadRequest
    ) -> UploadResult:
        """Check, store and enqueue an upload (or report it `unchanged`)."""
        if len(request.data) > self._cfg.max_file_bytes:
            raise TooLargeError(f"file exceeds {self._cfg.max_file_bytes} bytes")
        if not request.data:
            raise ValidationError("file", "must not be empty")
        external_id = (
            None if request.external_id is None else _validate_external_id(request.external_id)
        )
        if request.title is not None:
            validate_title(request.title)
        if request.metadata is not None:
            validate_metadata(request.metadata)
        collection = await self._d.collections.get(tenant, collection_id)
        inspected = await self._d.inspector.inspect(request.data, request.filename)
        existing = (
            None
            if external_id is None
            else await self._d.documents.find(tenant, collection_id, external_id)
        )
        await self._check_quotas(tenant, len(request.data), new_document=existing is None)
        title = request.title
        if title is None:
            title = existing.title if existing is not None else request.filename
        metadata = request.metadata
        if metadata is None:
            metadata = dict(existing.metadata) if existing is not None else {}
        return await self._d.documents.upload(
            tenant,
            NewUpload(
                collection_id=collection.id,
                external_id=external_id,
                title=validate_title(title[:TITLE_MAX_CHARS]),
                metadata=metadata,
                media_type=inspected.media_type,
                source_sha256=hashlib.sha256(request.data).digest(),
                data=request.data,
                chunking_hash=collection.settings.chunking.settings_hash(),
                pages=inspected.pages,
                ingest_delay=request.ingest_delay,
            ),
        )

    async def _check_quotas(self, tenant: TenantId, size: int, *, new_document: bool) -> None:
        if self._d.tenants is None:
            return
        quotas = (await self._d.tenants.get_tenant(tenant)).quotas
        if quotas.max_upload_bytes_per_day:
            today = await self._d.documents.uploaded_bytes_today(tenant)
            quotas.check("max_upload_bytes_per_day", today, size)
        if quotas.max_documents and new_document:
            quotas.check("max_documents", await self._d.documents.count(tenant), 1)

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> DocumentInfo:
        """One document with its chunk count and versions."""
        record = await self._d.documents.get(tenant, collection_id, document_id)
        versions = await self._d.documents.versions(tenant, document_id)
        count = await self._d.index.current_chunk_count(tenant, document_id)
        return DocumentInfo(record, count, tuple(versions))

    async def list_page(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        after: DocumentId | None,
        limit: int,
    ) -> list[DocumentRecord]:
        """Documents of a collection ordered by id; 404 when the collection is absent."""
        await self._d.collections.get(tenant, collection_id)
        return await self._d.documents.list_page(tenant, collection_id, after, limit)

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> DocumentInfo:
        """PATCH metadata and/or the display title (the indexed title changes on re-upload)."""
        if metadata is not None:
            validate_metadata(metadata)
        if title is not None:
            validate_title(title)
        await self._d.documents.update(
            tenant, collection_id, document_id, metadata=metadata, title=title
        )
        return await self.get(tenant, collection_id, document_id)

    async def delete(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> None:
        """Delete a document with all versions, chunks, vectors and blobs."""
        await self._d.documents.delete(tenant, collection_id, document_id)

    async def elements(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> list[Element]:
        """Elements of the current version; 409 `document_not_ready` before the first one."""
        record = await self._d.documents.get(tenant, collection_id, document_id)
        if record.current_version_id is None:
            raise ConflictError("the document has no indexed version yet", code=DOCUMENT_NOT_READY)
        return await self._d.documents.elements(tenant, record.current_version_id)

    async def text(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> str:
        """Current version text: elements joined by blank lines."""
        elements = await self.elements(tenant, collection_id, document_id)
        return "\n\n".join(e.text for e in elements)

    async def original(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        version: int | None = None,
    ) -> tuple[str, bytes]:
        """(media type, bytes) of the current or a given version (`original_not_retained`)."""
        if version is not None and version < 1:
            raise ValidationError("version", "must be >= 1")
        return await self._d.documents.original(tenant, collection_id, document_id, version)

    async def chunk(
        self, tenant: TenantId, collection_id: CollectionId, chunk_id: ChunkId
    ) -> ChunkDetail:
        """A chunk of a current version (`chunk_not_found`)."""
        return await self._d.chunks.chunk(tenant, collection_id, chunk_id)
