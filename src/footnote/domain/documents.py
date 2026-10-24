# SPDX-License-Identifier: Apache-2.0
"""Documents, versions, elements, chunks and memberships."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, NewType
from uuid import UUID

from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.tenancy import TenantId

CollectionId = NewType("CollectionId", UUID)
DocumentId = NewType("DocumentId", UUID)
VersionId = NewType("VersionId", UUID)
ChunkId = NewType("ChunkId", UUID)
JobId = NewType("JobId", UUID)


class ElementKind(StrEnum):
    """Structured piece of a parsed document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class Element:
    kind: ElementKind
    text: str
    order: int
    level: int | None = None
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    page_to: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "level": self.level,
            "heading_path": list(self.heading_path),
            "page": self.page,
            "page_to": self.page_to,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Element:
        return cls(
            kind=ElementKind(data["kind"]),
            text=str(data["text"]),
            order=int(data["order"]),
            level=data.get("level"),
            heading_path=tuple(data.get("heading_path") or ()),
            page=data.get("page"),
            page_to=data.get("page_to"),
        )


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    text: str
    heading_path: tuple[str, ...]
    embedded_text: str
    content_hash: str
    token_count: int
    page_from: int | None
    page_to: int | None
    oversize: bool = False
    overlap_chars: int = 0


class DocumentStatus(StrEnum):
    """Externally visible document status."""

    QUEUED = "queued"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class VersionStatus(StrEnum):
    """Status of one document version."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class Membership:
    version_id: VersionId
    ordinal: int
    chunk_id: ChunkId
    page_from: int | None
    page_to: int | None


@dataclass(frozen=True, slots=True)
class DocumentRef:
    id: DocumentId
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    id: CollectionId
    tenant_id: TenantId
    name: str
    settings: CollectionSettings
    active_space_id: int
    target_space_id: int | None
    chunk_count: int
    reembed: dict[str, Any] | None
    webhook_url: str | None
    webhook_secret: str | None
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NewCollection:
    name: str
    settings: CollectionSettings
    space_id: int
    webhook_url: str | None = None
    webhook_secret: str | None = None


@dataclass(frozen=True, slots=True)
class NewUpload:
    collection_id: CollectionId
    external_id: str | None
    title: str
    metadata: dict[str, Any]
    media_type: str
    source_sha256: bytes
    data: bytes
    chunking_hash: str
    pages: int | None = None
    ingest_delay: timedelta = timedelta(0)


@dataclass(frozen=True, slots=True)
class UploadResult:
    document_id: DocumentId
    unchanged: bool
    version_id: VersionId | None = None
    version: int | None = None
    job_id: JobId | None = None
    upload_bytes_today: int = 0


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: DocumentId
    collection_id: CollectionId
    external_id: str
    title: str
    metadata: dict[str, Any]
    status: DocumentStatus
    error: dict[str, Any] | None
    current_version_id: VersionId | None
    current_version: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class VersionRecord:
    id: VersionId
    document_id: DocumentId
    collection_id: CollectionId
    version: int
    source_sha256: bytes
    media_type: str
    bytes: int
    pages: int | None
    indexed_title: str
    chunking_hash: str
    status: VersionStatus
    error: dict[str, Any] | None
    has_original: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChunkBatch:
    collection_id: CollectionId
    document_id: DocumentId
    title: str
    language: Language
    drafts: Sequence[ChunkDraft]


@dataclass(frozen=True, slots=True)
class SwapResult:
    current: bool
    chunk_count: int
    job_ids: tuple[JobId, ...]


@dataclass(frozen=True, slots=True)
class GcStats:
    memberships: int
    chunks: int
    blobs: int


@dataclass(frozen=True, slots=True)
class InspectedFile:
    media_type: str
    pages: int | None = None


@dataclass(frozen=True, slots=True)
class ChunkPlacement:
    version: int
    ordinal: int
    page_from: int | None
    page_to: int | None


@dataclass(frozen=True, slots=True)
class ChunkDetail:
    id: ChunkId
    collection_id: CollectionId
    document_id: DocumentId
    document_title: str
    text: str
    heading_path: tuple[str, ...]
    token_count: int
    memberships: tuple[ChunkPlacement, ...]
