# SPDX-License-Identifier: Apache-2.0
"""Webhook delivery records and the event payload."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from footnote.domain.documents import CollectionId, DocumentId
from footnote.domain.webhooks import WebhookEvent

type DeliveryStatus = Literal["pending", "delivered", "failed"]


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    id: UUID
    collection_id: CollectionId
    event: str
    url: str
    status: str
    attempts: int
    response_code: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    delivery_id: UUID
    collection_id: CollectionId
    event: str
    url: str
    status: DeliveryStatus
    response_code: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentEvent:
    event: WebhookEvent
    collection_id: CollectionId
    document_id: DocumentId
    version: int
    status: str
    error: dict[str, Any] | None
    occurred_at: float

    def payload(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "collection_id": str(self.collection_id),
            "document_id": str(self.document_id),
            "version": self.version,
            "status": self.status,
            "error": self.error,
            "occurred_at": datetime.fromtimestamp(self.occurred_at, UTC).isoformat(),
        }
