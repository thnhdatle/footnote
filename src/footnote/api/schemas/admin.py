# SPDX-License-Identifier: Apache-2.0
"""Schemas of the admin routes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from footnote.api.schemas.common import Request
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec
from footnote.domain.tenancy import ApiKeyRecord, IssuedKey, Quotas, Tenant, TenantUsage


class QuotasIn(Request):
    """Per-tenant limits; 0 = unlimited; omitted fields keep their value on PATCH."""

    max_collections: int | None = Field(default=None, ge=0)
    max_documents: int | None = Field(default=None, ge=0)
    max_chunks: int | None = Field(default=None, ge=0)
    max_upload_bytes_per_day: int | None = Field(default=None, ge=0)

    def changes(self) -> dict[str, int]:
        return {k: v for k, v in self.model_dump(exclude_unset=True).items() if v is not None}


class QuotasOut(BaseModel):
    """Effective quotas (0 = unlimited)."""

    max_collections: int
    max_documents: int
    max_chunks: int
    max_upload_bytes_per_day: int

    @classmethod
    def of(cls, q: Quotas) -> QuotasOut:
        return cls(
            max_collections=q.max_collections,
            max_documents=q.max_documents,
            max_chunks=q.max_chunks,
            max_upload_bytes_per_day=q.max_upload_bytes_per_day,
        )


class TenantCreate(Request):
    """`POST /v1/admin/tenants`; quotas default to `tenancy.default_quotas`."""

    name: str = Field(min_length=1, max_length=256)
    quotas: QuotasIn | None = None


class TenantPatch(Request):
    """`PATCH /v1/admin/tenants/{id}`."""

    name: str | None = Field(default=None, min_length=1, max_length=256)
    quotas: QuotasIn | None = None


class TenantOut(BaseModel):
    """A tenant."""

    id: UUID
    name: str
    status: str
    quotas: QuotasOut

    @classmethod
    def of(cls, t: Tenant) -> TenantOut:
        return cls(id=t.id, name=t.name, status=t.status.value, quotas=QuotasOut.of(t.quotas))


class UsageOut(BaseModel):
    """`GET /v1/admin/tenants/{id}/usage`."""

    collections: int
    documents: int
    chunks: int
    vectors: int
    stored_bytes: int

    @classmethod
    def of(cls, u: TenantUsage) -> UsageOut:
        return cls(
            collections=u.collections,
            documents=u.documents,
            chunks=u.chunks,
            vectors=u.vectors,
            stored_bytes=u.stored_bytes,
        )


class KeyCreate(Request):
    """`POST /v1/admin/tenants/{id}/keys`."""

    name: str = Field(default="default", min_length=1, max_length=256)
    expires_at: datetime | None = None


class KeyOut(BaseModel):
    """A key without its secret."""

    id: UUID
    tenant_id: UUID
    prefix: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    expires_at: datetime | None
    revoked: bool

    @classmethod
    def of(cls, k: ApiKeyRecord) -> KeyOut:
        return cls(
            id=k.id,
            tenant_id=k.tenant_id,
            prefix=k.prefix,
            name=k.name,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            revoked_at=k.revoked_at,
            expires_at=k.expires_at,
            revoked=k.revoked_at is not None,
        )


class IssuedKeyOut(KeyOut):
    """A new key; `key` is shown once and never stored."""

    key: str

    @classmethod
    def of_issued(cls, issued: IssuedKey) -> IssuedKeyOut:
        return cls(**KeyOut.of(issued.record).model_dump(), key=issued.key)


class SpaceIn(Request):
    """`POST /v1/admin/embedding-spaces`; idempotent by name."""

    name: str = Field(min_length=1, max_length=63)
    provider: Literal["onnx_local", "openai_compat"]
    model: str = Field(min_length=1, max_length=256)
    dimensions: int | None = Field(
        default=None, ge=1, description="Required unless the manifest knows the model"
    )
    send_dimensions: bool = False
    normalise: bool = True
    calibrated_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    def spec(self) -> SpaceSpec:
        return SpaceSpec(
            name=self.name,
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions or 0,
            send_dimensions=self.send_dimensions,
            normalise=self.normalise,
            calibrated_threshold=self.calibrated_threshold,
        )


class SpaceOut(BaseModel):
    """A registered embedding space."""

    id: int
    name: str
    provider: str
    model: str
    dimensions: int
    send_dimensions: bool
    normalise: bool
    index_type: str
    calibrated_threshold: float | None

    @classmethod
    def of(cls, s: EmbeddingSpace) -> SpaceOut:
        return cls(
            id=s.id,
            name=s.name,
            provider=s.provider,
            model=s.model,
            dimensions=s.dimensions,
            send_dimensions=s.send_dimensions,
            normalise=s.normalise,
            index_type=s.index_type.value,
            calibrated_threshold=s.calibrated_threshold,
        )
