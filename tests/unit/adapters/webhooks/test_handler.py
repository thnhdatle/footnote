# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import orjson
import pytest

from footnote.adapters.queue import JobContext
from footnote.adapters.webhooks import WebhookConfig, WebhookJobHandler, WebhookSender
from footnote.domain.deliveries import AttemptRecord
from footnote.domain.documents import CollectionId, CollectionRecord, JobId
from footnote.domain.errors import NotFoundError
from footnote.domain.jobs import ClaimedJob, JobRecord, PermanentJobError
from footnote.domain.settings import CollectionSettings
from footnote.domain.tenancy import TenantId
from footnote.domain.webhooks import verify_signature

TENANT = TenantId(UUID(int=1))
COLLECTION = CollectionId(UUID(int=2))
NOW = datetime(2026, 1, 1)
ARGS: dict[str, Any] = {
    "collection_id": str(COLLECTION),
    "event": "document.ready",
    "document_id": str(UUID(int=3)),
    "version": 2,
    "status": "ready",
    "error": None,
    "occurred_at": "2026-01-01T00:00:00Z",
}


def _collection(**overrides: Any) -> CollectionRecord:
    base = CollectionRecord(
        COLLECTION, TENANT, "docs", CollectionSettings(), 1, None, 0, None,
        "https://hooks.example.com/in", "s3cret", "active", NOW, NOW,
    )  # fmt: skip
    return replace(base, **overrides)


@dataclass
class Fakes:
    collection: CollectionRecord | None = field(default_factory=_collection)
    attempts_made: int = 0
    recorded: list[AttemptRecord] = field(default_factory=list)

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int:
        return self.attempts_made

    async def record_attempt(self, tenant: TenantId, attempt: AttemptRecord) -> None:
        self.recorded.append(attempt)


JOB = ClaimedJob(JobId(uuid4()), TENANT, "webhook")


class Jobs:
    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        return JobRecord(
            job_id, tenant, "webhook", ARGS, "running", 1, None, None, NOW, NOW, None, None, NOW
        )  # fmt: skip


class Collections:
    def __init__(self, fakes: Fakes) -> None:
        self._fakes = fakes

    async def get(self, tenant: TenantId, collection_id: CollectionId) -> CollectionRecord:
        if self._fakes.collection is None:
            raise NotFoundError("collection not found")
        return self._fakes.collection


def _handler(
    fakes: Fakes, respond: httpx.Response | None = None, *, max_attempts: int = 3
) -> tuple[WebhookJobHandler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond if respond is not None else httpx.Response(200)

    async def resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    sender = WebhookSender(
        httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        WebhookConfig(max_attempts=max_attempts),
        resolve,
    )
    handler = WebhookJobHandler(Jobs(), Collections(fakes), fakes, sender, clock=lambda: 1234.9)
    return handler, seen


def _ctx() -> JobContext:
    class NoStore:
        pass

    return JobContext(JOB, NoStore(), "w")  # type: ignore[arg-type]


async def test_delivers_the_f12_payload_signed() -> None:
    fakes = Fakes()
    handler, seen = _handler(fakes)
    await handler(_ctx())
    request = seen[0]
    assert orjson.loads(request.content) == ARGS
    assert verify_signature(
        b"s3cret", 1234, request.content, request.headers["x-footnote-signature"]
    )
    (attempt,) = fakes.recorded
    assert (attempt.status, attempt.response_code, attempt.event) == (
        "delivered",
        200,
        "document.ready",
    )
    assert attempt.delivery_id == JOB.id


async def test_failure_before_last_attempt_is_pending_and_retried() -> None:
    fakes = Fakes(attempts_made=1)
    handler, _ = _handler(fakes, httpx.Response(500))
    with pytest.raises(RuntimeError, match="status 500"):
        await handler(_ctx())
    assert fakes.recorded[0].status == "pending"
    assert fakes.recorded[0].response_code == 500


async def test_last_attempt_fails_permanently() -> None:
    fakes = Fakes(attempts_made=2)
    handler, _ = _handler(fakes, httpx.Response(500), max_attempts=3)
    with pytest.raises(PermanentJobError):
        await handler(_ctx())
    assert fakes.recorded[0].status == "failed"


async def test_policy_rejection_is_permanent_and_recorded() -> None:
    fakes = Fakes(collection=_collection(webhook_url="http://hooks.example.com/"))
    handler, seen = _handler(fakes)
    with pytest.raises(PermanentJobError):
        await handler(_ctx())
    assert seen == []
    assert fakes.recorded[0].status == "failed"


@pytest.mark.parametrize(
    "collection",
    [None, _collection(webhook_url=None), _collection(webhook_secret=None)],
    ids=["deleted", "no-url", "no-secret"],
)
async def test_missing_collection_or_webhook_is_permanent(
    collection: CollectionRecord | None,
) -> None:
    fakes = Fakes(collection=collection)
    handler, seen = _handler(fakes)
    with pytest.raises(PermanentJobError):
        await handler(_ctx())
    assert seen == []
    assert fakes.recorded == []
