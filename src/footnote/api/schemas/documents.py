# SPDX-License-Identifier: Apache-2.0
"""Document, version, element and chunk models."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from footnote.api.schemas.common import Request
from footnote.app.documents import DocumentInfo
from footnote.domain.documents import (
    ChunkDetail,
    DocumentRecord,
    DocumentStatus,
    Element,
    ElementKind,
    UploadResult,
    VersionRecord,
    VersionStatus,
)


class UploadOut(BaseModel):
    """Upload result: 202 with a job, or 200 `unchanged` (no version, metadata updated)."""

    document_id: UUID
    unchanged: bool
    version: int | None = None
    job_id: UUID | None = None

    @classmethod
    def of(cls, r: UploadResult) -> UploadOut:
        """Build from the app-layer result."""
        return cls(
            document_id=r.document_id, unchanged=r.unchanged, version=r.version, job_id=r.job_id
        )


class DocumentErrorOut(BaseModel):
    """Why a document failed (`parse_failed`, `quota_exceeded`, …)."""

    code: str
    message: str


class VersionOut(BaseModel):
    """One document version."""

    version: int
    status: VersionStatus
    media_type: str
    bytes: int
    pages: int | None
    sha256: str
    indexed_title: str
    error: DocumentErrorOut | None
    has_original: bool
    created_at: datetime

    @classmethod
    def of(cls, v: VersionRecord) -> VersionOut:
        """Build from the domain record."""
        return cls(
            version=v.version,
            status=v.status,
            media_type=v.media_type,
            bytes=v.bytes,
            pages=v.pages,
            sha256=v.source_sha256.hex(),
            indexed_title=v.indexed_title,
            error=_error(v.error),
            has_original=v.has_original,
            created_at=v.created_at,
        )


def _error(error: dict[str, Any] | None) -> DocumentErrorOut | None:
    if not error:
        return None
    return DocumentErrorOut(
        code=str(error.get("code", "internal")), message=str(error.get("message", ""))
    )


class DocumentOut(BaseModel):
    """A document."""

    id: UUID
    collection_id: UUID
    external_id: str
    title: str
    metadata: dict[str, Any]
    status: DocumentStatus
    error: DocumentErrorOut | None
    current_version: int | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, d: DocumentRecord) -> DocumentOut:
        """Build from the domain record."""
        return cls(
            id=d.id,
            collection_id=d.collection_id,
            external_id=d.external_id,
            title=d.title,
            metadata=d.metadata,
            status=d.status,
            error=_error(d.error),
            current_version=d.current_version,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )


class DocumentDetailOut(DocumentOut):
    """A document with its chunk count and versions."""

    chunk_count: int
    versions: list[VersionOut]

    @classmethod
    def of_info(cls, info: DocumentInfo) -> DocumentDetailOut:
        """Build from the app-layer view."""
        return cls(
            **DocumentOut.of(info.record).model_dump(),
            chunk_count=info.chunk_count,
            versions=[VersionOut.of(v) for v in info.versions],
        )


class DocumentPatch(Request):
    """`PATCH …/documents/{doc_id}`: filters see new metadata at once; the indexed title
    changes with the next version."""

    metadata: dict[str, Any] | None = Field(default=None, description="JSON object <= 16 KiB")
    title: str | None = Field(default=None, max_length=512)


class ElementOut(BaseModel):
    """A parsed element."""

    kind: ElementKind
    text: str
    level: int | None
    heading_path: list[str]
    page: int | None
    page_to: int | None
    order: int

    @classmethod
    def of(cls, e: Element) -> ElementOut:
        """Build from the domain element."""
        return cls(
            kind=e.kind,
            text=e.text,
            level=e.level,
            heading_path=list(e.heading_path),
            page=e.page,
            page_to=e.page_to,
            order=e.order,
        )


class ElementsOut(BaseModel):
    """`?format=elements` body."""

    data: list[ElementOut]


class ChunkMembershipOut(BaseModel):
    """A current-version membership."""

    version: int
    ordinal: int
    page_from: int | None
    page_to: int | None


class ChunkOut(BaseModel):
    """`GET /v1/collections/{id}/chunks/{chunk_id}`."""

    id: UUID
    collection_id: UUID
    document_id: UUID
    document_title: str
    text: str
    heading_path: list[str]
    token_count: int
    memberships: list[ChunkMembershipOut]

    @classmethod
    def of(cls, c: ChunkDetail) -> ChunkOut:
        """Build from the domain record."""
        return cls(
            id=c.id,
            collection_id=c.collection_id,
            document_id=c.document_id,
            document_title=c.document_title,
            text=c.text,
            heading_path=list(c.heading_path),
            token_count=c.token_count,
            memberships=[
                ChunkMembershipOut(
                    version=m.version, ordinal=m.ordinal, page_from=m.page_from, page_to=m.page_to
                )
                for m in c.memberships
            ],
        )
