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
    """Lifecycle of a tenant."""

    ACTIVE = "active"
    DELETING = "deleting"


@dataclass(frozen=True, slots=True)
class Quotas:
    """Per-tenant limits; 0 means unlimited."""

    max_collections: int = 0
    max_documents: int = 0
    max_chunks: int = 0
    max_upload_bytes_per_day: int = 0

    def validate(self) -> None:
        """Reject negative limits."""
        for name in ("max_collections", "max_documents", "max_chunks", "max_upload_bytes_per_day"):
            if getattr(self, name) < 0:
                raise ValidationError(f"quotas.{name}", "must be >= 0")

    def check(self, name: str, current: int, adding: int) -> None:
        """Raise QuotaExceededError when `current + adding` exceeds the named limit."""
        limit = int(getattr(self, name))
        if limit and current + adding > limit:
            raise QuotaExceededError(f"{name} {limit} would be exceeded ({current} + {adding})")


@dataclass(frozen=True, slots=True)
class Tenant:
    """Isolation boundary."""

    id: TenantId
    name: str
    quotas: Quotas
    status: TenantStatus = TenantStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    """A stored key without its secret."""

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
    """A newly created key; `key` is shown once and never stored."""

    record: ApiKeyRecord
    key: str


@dataclass(frozen=True, slots=True)
class TenantUsage:
    """Admin usage counters of one tenant."""

    collections: int
    documents: int
    chunks: int
    vectors: int
    stored_bytes: int


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """One line of a tenant export: `{"type": …, "data": {…}}`."""

    type: str
    data: dict[str, object]
