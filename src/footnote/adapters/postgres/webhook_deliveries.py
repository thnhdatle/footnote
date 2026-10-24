# SPDX-License-Identifier: Apache-2.0
"""Webhook delivery log."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from psycopg import sql

from footnote.adapters.postgres.session import Database, fetch_all, fetch_one
from footnote.domain.deliveries import AttemptRecord, DeliveryRecord
from footnote.domain.documents import CollectionId
from footnote.domain.tenancy import TenantId

_MAX_ERROR_CHARS = 1000


_COLS = (
    "d.id, d.collection_id, d.event, d.url, d.status, d.attempts, d.response_code, d.error, "
    "d.created_at, d.updated_at, d.delivered_at"
)


def _record(row: dict[str, Any]) -> DeliveryRecord:
    return DeliveryRecord(
        id=row["id"],
        collection_id=CollectionId(row["collection_id"]),
        event=row["event"],
        url=row["url"],
        status=row["status"],
        attempts=int(row["attempts"]),
        response_code=row["response_code"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        delivered_at=row["delivered_at"],
    )


class PgWebhookDeliveries:
    """Stores webhook delivery attempts."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int:
        query = sql.SQL(
            "SELECT d.attempts FROM footnote.webhook_deliveries d WHERE {} AND d.id = %(id)s"
        ).format(self._db.tenant_filter("d"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "id": delivery_id})
        return int(row["attempts"]) if row else 0

    async def record_attempt(self, tenant: TenantId, attempt: AttemptRecord) -> None:
        """Records one delivery attempt."""
        error = attempt.error[:_MAX_ERROR_CHARS] if attempt.error is not None else None
        async with self._db.tenant_tx(tenant) as conn:
            await conn.execute(
                "INSERT INTO footnote.webhook_deliveries (tenant_id, id, collection_id, event, "
                "url, status, attempts, response_code, error, delivered_at) "
                "VALUES (%(tenant)s, %(id)s, %(cid)s, %(event)s, %(url)s, %(status)s, 1, "
                "%(code)s, %(error)s, CASE WHEN %(status)s = 'delivered' THEN now() END) "
                "ON CONFLICT (tenant_id, id) DO UPDATE SET status = EXCLUDED.status, "
                "attempts = footnote.webhook_deliveries.attempts + 1, "
                "response_code = EXCLUDED.response_code, error = EXCLUDED.error, "
                "url = EXCLUDED.url, updated_at = now(), delivered_at = EXCLUDED.delivered_at",
                {
                    "tenant": tenant,
                    "id": attempt.delivery_id,
                    "cid": attempt.collection_id,
                    "event": attempt.event,
                    "url": attempt.url,
                    "status": attempt.status,
                    "code": attempt.response_code,
                    "error": error,
                },
            )

    async def list_for_collection(
        self, tenant: TenantId, collection_id: CollectionId, limit: int = 50
    ) -> list[DeliveryRecord]:
        query = sql.SQL(
            "SELECT {} FROM footnote.webhook_deliveries d WHERE {} AND d.collection_id = %(cid)s "
            "ORDER BY d.created_at DESC, d.id DESC LIMIT %(limit)s"
        ).format(sql.SQL(_COLS), self._db.tenant_filter("d"))
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(
                conn, query, {"tenant": tenant, "cid": collection_id, "limit": limit}
            )
        return [_record(r) for r in rows]


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_webhookdeliverylog(db: Database) -> ports.WebhookDeliveryLog:
        return PgWebhookDeliveries(db)
