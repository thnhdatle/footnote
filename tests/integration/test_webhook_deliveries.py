# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
import orjson
import pytest
import pytest_asyncio

from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database, Pool
from footnote.adapters.postgres.webhook_deliveries import PgWebhookDeliveries
from footnote.adapters.queue import Worker, WorkerConfig
from footnote.adapters.webhooks import WebhookConfig, WebhookJobHandler, WebhookSender
from footnote.domain.deliveries import AttemptRecord
from footnote.domain.documents import JobId
from footnote.domain.errors import NotFoundError
from footnote.domain.jobs import NewJob
from footnote.domain.webhooks import verify_signature
from tests.integration.world import World, seed

pytestmark = pytest.mark.integration

URL = "https://example.com/hook"


@pytest_asyncio.fixture
async def a(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, f"wa-{uuid.uuid4().hex[:8]}", ["alpha text"])


@pytest_asyncio.fixture
async def b(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, f"wb-{uuid.uuid4().hex[:8]}", ["beta text"])


def _attempt(world: World, delivery_id: uuid.UUID, status: str, code: int | None) -> AttemptRecord:
    return AttemptRecord(
        delivery_id,
        world.collection.id,
        "document.ready",
        URL,
        status,
        code,
        None,  # type: ignore[arg-type]
    )


async def test_attempts_accumulate_on_one_row(app_db: Database, a: World) -> None:
    log = PgWebhookDeliveries(app_db)
    delivery = uuid.uuid4()
    assert await log.attempts(a.tenant, delivery) == 0
    await log.record_attempt(a.tenant, _attempt(a, delivery, "pending", 500))
    await log.record_attempt(a.tenant, _attempt(a, delivery, "delivered", 200))
    assert await log.attempts(a.tenant, delivery) == 2
    rows = {r.id: r for r in await log.list_for_collection(a.tenant, a.collection.id)}
    row = rows[delivery]
    assert (row.status, row.attempts, row.response_code) == ("delivered", 2, 200)
    assert row.delivered_at is not None
    assert row.url == URL


async def test_failed_delivery_keeps_the_last_error(app_db: Database, a: World) -> None:
    log = PgWebhookDeliveries(app_db)
    delivery = uuid.uuid4()
    failed = AttemptRecord(
        delivery, a.collection.id, "document.failed", URL, "failed", 503, "x" * 5000
    )
    await log.record_attempt(a.tenant, failed)
    (row,) = [
        r for r in await log.list_for_collection(a.tenant, a.collection.id) if r.id == delivery
    ]
    assert row.status == "failed"
    assert row.delivered_at is None
    assert row.error is not None
    assert len(row.error) == 1000


async def test_list_is_newest_first_and_limited(app_db: Database, a: World) -> None:
    log = PgWebhookDeliveries(app_db)
    ids = [uuid.uuid4() for _ in range(3)]
    for delivery in ids:
        await log.record_attempt(a.tenant, _attempt(a, delivery, "delivered", 200))
    page = await log.list_for_collection(a.tenant, a.collection.id, limit=2)
    assert [r.id for r in page] == [ids[2], ids[1]]


async def test_cross_tenant_isolation(app_pool: Pool, a: World, b: World) -> None:
    for predicates in (True, False):
        intruder = PgWebhookDeliveries(Database(app_pool, app_tenant_predicates=predicates))
        assert await intruder.list_for_collection(b.tenant, a.collection.id) == []
        delivery = uuid.uuid4()
        await PgWebhookDeliveries(Database(app_pool)).record_attempt(
            a.tenant, _attempt(a, delivery, "delivered", 200)
        )
        assert await intruder.attempts(b.tenant, delivery) == 0
    with pytest.raises(NotFoundError):
        await PgWebhookDeliveries(Database(app_pool)).record_attempt(
            b.tenant, _attempt(a, uuid.uuid4(), "delivered", 200)
        )


async def _run_webhook_job(
    app_db: Database, world: World, respond: list[httpx.Response], max_attempts: int = 3
) -> tuple[JobId, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond[min(len(seen), len(respond)) - 1]

    async def resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    jobs = PgJobs(app_db)
    sender = WebhookSender(
        httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        WebhookConfig(max_attempts=max_attempts),
        resolve,
    )
    handler = WebhookJobHandler(jobs, PgCollections(app_db), PgWebhookDeliveries(app_db), sender)
    args = {
        "collection_id": str(world.collection.id),
        "event": "document.ready",
        "document_id": str(world.document_id),
        "version": 1,
        "status": "ready",
        "error": None,
        "occurred_at": "2026-01-01T00:00:00Z",
    }
    job_id = await jobs.enqueue(world.tenant, NewJob("webhook", args))
    worker = Worker(
        jobs,
        {"webhook": handler},
        WorkerConfig("w-hook", poll_interval=0.05, backoff_base=0.05, backoff_cap=0.1),
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        async with asyncio.timeout(20):
            while (await jobs.get(world.tenant, job_id)).status not in {"done", "failed"}:
                await asyncio.sleep(0.02)
    finally:
        stop.set()
        await asyncio.wait_for(task, 20)
    return job_id, seen


async def test_webhook_job_delivers_signed_payload(app_db: Database, a: World) -> None:
    job_id, seen = await _run_webhook_job(app_db, a, [httpx.Response(200)])
    assert (await PgJobs(app_db).get(a.tenant, job_id)).status == "done"
    request = seen[0]
    assert orjson.loads(request.content)["event"] == "document.ready"
    ts = int(request.headers["x-footnote-timestamp"])
    assert verify_signature(b"s3cret", ts, request.content, request.headers["x-footnote-signature"])
    rows = {
        r.id: r
        for r in await PgWebhookDeliveries(app_db).list_for_collection(a.tenant, a.collection.id)
    }
    assert (rows[job_id].status, rows[job_id].attempts) == ("delivered", 1)


async def test_webhook_job_retries_then_succeeds(app_db: Database, a: World) -> None:
    job_id, seen = await _run_webhook_job(app_db, a, [httpx.Response(500), httpx.Response(204)])
    assert len(seen) == 2
    rows = {
        r.id: r
        for r in await PgWebhookDeliveries(app_db).list_for_collection(a.tenant, a.collection.id)
    }
    assert (rows[job_id].status, rows[job_id].attempts, rows[job_id].response_code) == (
        "delivered",
        2,
        204,
    )


async def test_webhook_job_gives_up_after_max_attempts(app_db: Database, a: World) -> None:
    job_id, seen = await _run_webhook_job(app_db, a, [httpx.Response(500)], max_attempts=3)
    assert len(seen) == 3
    assert (await PgJobs(app_db).get(a.tenant, job_id)).status == "failed"
    rows = {
        r.id: r
        for r in await PgWebhookDeliveries(app_db).list_for_collection(a.tenant, a.collection.id)
    }
    assert (rows[job_id].status, rows[job_id].attempts) == ("failed", 3)
