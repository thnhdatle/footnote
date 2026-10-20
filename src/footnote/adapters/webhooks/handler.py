# SPDX-License-Identifier: Apache-2.0
"""The `webhook` job handler: build the payload, deliver, record the attempt.

Job args: `collection_id`, `event`, `document_id`, `version`, `status`, `error`,
`occurred_at` (the enqueuer fills them; the handler adds nothing but the signature).
`on_attempt(status)` reports each recorded attempt (`delivered`, `pending`, `failed`) for
`footnote_webhook_deliveries_total`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

import orjson

from footnote.adapters.queue.worker import JobContext
from footnote.adapters.webhooks.sender import WebhookSender
from footnote.domain.deliveries import AttemptRecord, DeliveryStatus
from footnote.domain.documents import CollectionId, CollectionRecord, JobId
from footnote.domain.errors import NotFoundError
from footnote.domain.jobs import JobRecord, PermanentJobError
from footnote.domain.tenancy import TenantId

_PAYLOAD_KEYS = (
    "event",
    "collection_id",
    "document_id",
    "version",
    "status",
    "error",
    "occurred_at",
)


class JobReader(Protocol):
    """Reads a job row (`PgJobs` implements it)."""

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        """One job of the tenant."""


class CollectionReader(Protocol):
    """Reads a collection with its webhook settings (`PgCollections` implements it)."""

    async def get(self, tenant: TenantId, collection_id: CollectionId) -> CollectionRecord:
        """One collection; NotFoundError when absent."""


class DeliveryLog(Protocol):
    """The delivery log (`PgWebhookDeliveries` implements it)."""

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int:
        """Attempts recorded so far."""

    async def record_attempt(self, tenant: TenantId, attempt: AttemptRecord) -> None:
        """Record one attempt."""


class WebhookJobHandler:
    """Callable `Handler` for kind `webhook`; retries are driven by the queue worker."""

    def __init__(  # noqa: PLR0913 - optional clock and metrics observer
        self,
        jobs: JobReader,
        collections: CollectionReader,
        deliveries: DeliveryLog,
        sender: WebhookSender,
        clock: Callable[[], float] = time.time,
        *,
        on_attempt: Callable[[str], None] | None = None,
    ) -> None:
        self._jobs = jobs
        self._collections = collections
        self._deliveries = deliveries
        self._sender = sender
        self._clock = clock
        self._on_attempt = on_attempt

    async def __call__(self, ctx: JobContext) -> None:
        """Deliver once; raise so the worker retries, or `PermanentJobError` to stop."""
        tenant = ctx.job.tenant_id
        record: JobRecord = await self._jobs.get(tenant, ctx.job.id)
        args = record.args
        collection_id = CollectionId(UUID(str(args["collection_id"])))
        try:
            collection = await self._collections.get(tenant, collection_id)
        except NotFoundError as exc:
            raise PermanentJobError("collection no longer exists") from exc
        if not collection.webhook_url or not collection.webhook_secret:
            raise PermanentJobError("collection has no webhook configured")
        body = orjson.dumps({k: args.get(k) for k in _PAYLOAD_KEYS})
        result = await self._sender.send(
            collection.webhook_url,
            collection.webhook_secret.encode(),
            body,
            int(self._clock()),
        )
        previous = await self._deliveries.attempts(tenant, ctx.job.id)
        final = result.delivered or result.permanent or previous + 1 >= self._sender.max_attempts
        status: DeliveryStatus = (
            "delivered" if result.delivered else "failed" if final else "pending"
        )
        await self._deliveries.record_attempt(
            tenant,
            AttemptRecord(
                ctx.job.id,
                collection_id,
                str(args["event"]),
                collection.webhook_url,
                status,
                result.status_code,
                result.error,
            ),
        )
        if self._on_attempt is not None:
            self._on_attempt(status)
        if result.delivered:
            return
        message = result.error or "delivery failed"
        if final:
            raise PermanentJobError(message)
        raise RuntimeError(message)
