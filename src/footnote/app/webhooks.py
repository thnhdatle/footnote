# SPDX-License-Identifier: Apache-2.0
"""Turns document outcomes into `webhook` jobs."""

from __future__ import annotations

from typing import Any

from footnote.domain.deliveries import DocumentEvent
from footnote.domain.documents import CollectionRecord, JobId, VersionRecord
from footnote.domain.jobs import JobKind, NewJob
from footnote.domain.ports import Clock, JobQueue
from footnote.domain.tenancy import TenantId
from footnote.domain.webhooks import WebhookEvent

_STATUS = {WebhookEvent.DOCUMENT_READY: "ready", WebhookEvent.DOCUMENT_FAILED: "failed"}


class WebhookNotifier:
    """Builds and queues `webhook` jobs for subscribed events."""

    def __init__(self, jobs: JobQueue, clock: Clock) -> None:
        self._jobs = jobs
        self._clock = clock

    def job_for(
        self,
        collection: CollectionRecord,
        event: WebhookEvent,
        version: VersionRecord,
        error: dict[str, Any] | None = None,
    ) -> NewJob | None:
        if not collection.webhook_url or not collection.webhook_secret:
            return None
        if not collection.settings.webhook.wants(event):
            return None
        payload = DocumentEvent(
            event=event,
            collection_id=collection.id,
            document_id=version.document_id,
            version=version.version,
            status=_STATUS[event],
            error=error,
            occurred_at=self._clock.now(),
        ).payload()
        return NewJob(JobKind.WEBHOOK.value, payload)

    async def notify(
        self,
        tenant: TenantId,
        collection: CollectionRecord,
        event: WebhookEvent,
        version: VersionRecord,
        *,
        error: dict[str, Any] | None = None,
    ) -> JobId | None:
        job = self.job_for(collection, event, version, error)
        if job is None:
            return None
        return await self._jobs.enqueue(tenant, job)
