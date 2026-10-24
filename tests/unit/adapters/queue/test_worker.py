# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from footnote.adapters.queue import JobContext, Worker, WorkerConfig
from footnote.domain.documents import JobId
from footnote.domain.jobs import ClaimedJob, JobRecord, PermanentJobError
from footnote.domain.tenancy import TenantId

TENANT = TenantId(UUID(int=1))
NOW = datetime(2026, 1, 1)


@dataclass
class FakeJob:
    claimed: ClaimedJob
    status: str = "queued"
    attempts: int = 0
    last_error: str | None = None
    retry_in: timedelta | None = None
    progress: tuple[int, int] | None = None
    extends: int = 0
    held: bool = False


@dataclass
class FakeStore:
    jobs: dict[JobId, FakeJob] = field(default_factory=dict)
    claim_calls: list[int] = field(default_factory=list)
    owned: bool = True
    claim_error: Exception | None = None

    def add(self, kind: str = "k") -> FakeJob:
        job = FakeJob(ClaimedJob(JobId(uuid4()), TENANT, kind))
        self.jobs[job.claimed.id] = job
        return job

    async def claim(self, worker: str, n: int, lease: timedelta) -> list[ClaimedJob]:
        self.claim_calls.append(n)
        if self.claim_error is not None:
            raise self.claim_error
        out = []
        for job in self.jobs.values():
            if job.status == "queued" and not job.held and len(out) < n:
                job.status = "running"
                job.attempts += 1
                out.append(job.claimed)
        return out

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        job = self.jobs[job_id]
        return JobRecord(
            job_id, tenant, job.claimed.kind, {}, job.status, job.attempts, job.last_error,
            None, NOW, NOW, None, None, NOW,
        )  # fmt: skip

    async def extend_lease(
        self, tenant: TenantId, job_id: JobId, worker: str, lease: timedelta
    ) -> bool:
        self.jobs[job_id].extends += 1
        return self.owned

    async def set_progress(
        self, tenant: TenantId, job_id: JobId, worker: str, done: int, total: int
    ) -> bool:
        self.jobs[job_id].progress = (done, total)
        return True

    async def complete(self, tenant: TenantId, job_id: JobId, worker: str) -> bool:
        self.jobs[job_id].status = "done"
        return True

    async def fail(
        self, tenant: TenantId, job_id: JobId, worker: str, error: str, retry_in: timedelta | None
    ) -> bool:
        job = self.jobs[job_id]
        job.last_error = error
        job.retry_in = retry_in
        job.status = "failed" if retry_in is None else "queued"
        job.held = retry_in is not None and retry_in > timedelta(0)
        return True


async def eventually(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.005)


@asynccontextmanager
async def running(
    store: FakeStore, handlers: dict[str, Any], **cfg: Any
) -> AsyncIterator[tuple[Worker, asyncio.Task[None], asyncio.Event]]:
    defaults: dict[str, Any] = {"poll_interval": 0.01, "shutdown_timeout": 1.0}
    config = WorkerConfig(worker_id="w", **{**defaults, **cfg})
    worker = Worker(store, handlers, config, rand=lambda: 1.0)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    try:
        yield worker, task, stop
    finally:
        stop.set()
        await asyncio.wait_for(task, 5)


async def test_successful_job_is_completed_with_progress() -> None:
    store = FakeStore()
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        await ctx.progress(3, 4)

    async with running(store, {"k": handler}):
        await eventually(lambda: job.status == "done")
    assert job.progress == (3, 4)


async def test_failure_requeues_with_jittered_backoff() -> None:
    store = FakeStore()
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        raise ValueError("boom")

    async with running(store, {"k": handler}, backoff_base=2.0):
        await eventually(lambda: job.last_error is not None)
    assert job.last_error == "ValueError: boom"
    assert job.retry_in == timedelta(seconds=2)


async def test_attempts_exhausted_fails_for_good() -> None:
    store = FakeStore()
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        raise ValueError("again")

    async with running(store, {"k": handler}, max_attempts=3, backoff_base=0.0):
        await eventually(lambda: job.status == "failed")
    assert job.attempts == 3
    assert job.retry_in is None


async def test_permanent_error_fails_immediately() -> None:
    store = FakeStore()
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        raise PermanentJobError("collection gone")

    async with running(store, {"k": handler}):
        await eventually(lambda: job.status == "failed")
    assert job.attempts == 1
    assert job.last_error == "collection gone"


async def test_unknown_kind_fails_permanently() -> None:
    store = FakeStore()
    job = store.add("mystery")
    async with running(store, {}):
        await eventually(lambda: job.status == "failed")
    assert job.last_error is not None
    assert "no handler" in job.last_error


async def test_error_text_is_truncated() -> None:
    store = FakeStore()
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        raise PermanentJobError("x" * 10_000)

    async with running(store, {"k": handler}):
        await eventually(lambda: job.status == "failed")
    assert job.last_error is not None
    assert len(job.last_error) == 2000


async def test_lease_is_renewed_while_running() -> None:
    store = FakeStore()
    job = store.add()
    release = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        await release.wait()

    async with running(store, {"k": handler}, lease=timedelta(milliseconds=30)):
        await eventually(lambda: job.extends >= 2)
        assert job.status == "running"
        release.set()
        await eventually(lambda: job.status == "done")


async def test_lost_lease_cancels_the_handler() -> None:
    store = FakeStore(owned=False)
    job = store.add()
    cancelled = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with running(store, {"k": handler}, lease=timedelta(milliseconds=30)):
        await asyncio.wait_for(cancelled.wait(), 5)
    assert job.status == "running"


async def test_concurrency_is_bounded_and_claims_only_free_slots() -> None:
    store = FakeStore()
    jobs = [store.add() for _ in range(5)]
    release = asyncio.Event()
    active = 0
    peak = 0

    async def handler(ctx: JobContext) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await release.wait()
        active -= 1

    async with running(store, {"k": handler}, concurrency=2):
        await eventually(lambda: active == 2)
        assert [j.status for j in jobs].count("running") == 2
        release.set()
        await eventually(lambda: all(j.status == "done" for j in jobs))
    assert peak == 2
    assert max(store.claim_calls) == 2


async def test_claim_errors_are_survived() -> None:
    store = FakeStore(claim_error=RuntimeError("db down"))
    job = store.add()

    async def handler(ctx: JobContext) -> None:
        return None

    async with running(store, {"k": handler}):
        await eventually(lambda: len(store.claim_calls) >= 2)
        store.claim_error = None
        await eventually(lambda: job.status == "done")


async def test_graceful_shutdown_waits_for_running_jobs() -> None:
    store = FakeStore()
    job = store.add()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        started.set()
        await release.wait()

    worker = Worker(
        store, {"k": handler}, WorkerConfig("w", poll_interval=0.01, shutdown_timeout=5)
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    await asyncio.wait_for(started.wait(), 5)
    stop.set()
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, 5)
    assert job.status == "done"


async def test_shutdown_timeout_cancels_and_leaves_the_lease() -> None:
    store = FakeStore()
    job = store.add()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = Worker(
        store, {"k": handler}, WorkerConfig("w", poll_interval=0.01, shutdown_timeout=0.05)
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    await asyncio.wait_for(started.wait(), 5)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert cancelled.is_set()
    assert job.status == "running"


def test_renew_interval_is_a_third_of_the_lease() -> None:
    assert WorkerConfig("w", lease=timedelta(seconds=60)).renew_interval == 20.0
