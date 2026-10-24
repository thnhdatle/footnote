# SPDX-License-Identifier: Apache-2.0
"""Tenants and quotas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType
from uuid import UUID

from footnote.domain.errors import QuotaExceededError, ValidationError

TenantId = NewType("TenantId", UUID)


class TenantStatus(StrEnum):
    ACTIVE = "active"
    DELETING = "deleting"


@dataclass(frozen=True, slots=True)
class Quotas:
    max_collections: int = 0
    max_documents: int = 0
    max_chunks: int = 0
    max_upload_bytes_per_day: int = 0

    def validate(self) -> None:
        for name in ("max_collections", "max_documents", "max_chunks", "max_upload_bytes_per_day"):
            if getattr(self, name) < 0:
                raise ValidationError(f"quotas.{name}", "must be >= 0")

    def check(self, name: str, current: int, adding: int) -> None:
        limit = int(getattr(self, name))
        if limit and current + adding > limit:
            raise QuotaExceededError(f"{name} {limit} would be exceeded ({current} + {adding})")


@dataclass(frozen=True, slots=True)
class Tenant:
    id: TenantId
    name: str
    quotas: Quotas
    status: TenantStatus = TenantStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: UUID
    tenant_id: TenantId
    prefix: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class IssuedKey:
    record: ApiKeyRecord
    key: str


@dataclass(frozen=True, slots=True)
class TenantUsage:
    collections: int
    documents: int
    chunks: int
    vectors: int
    stored_bytes: int


@dataclass(frozen=True, slots=True)
class ExportRecord:
    type: str
    data: dict[str, object]
