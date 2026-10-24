# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
import pytest_asyncio

from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database
from footnote.adapters.queue import JobContext, Worker, WorkerConfig
from footnote.domain.documents import JobId
from footnote.domain.jobs import JobRecord, NewJob, PermanentJobError
from tests.integration.world import World, seed

pytestmark = pytest.mark.integration

KIND = "reindex"


@pytest_asyncio.fixture
async def world(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, f"q-{uuid.uuid4().hex[:8]}", ["queue test text"])


async def _record(jobs: PgJobs, world: World, job_id: JobId) -> JobRecord:
    return await jobs.get(world.tenant, job_id)


async def _wait_status(jobs: PgJobs, world: World, job_id: JobId, status: str) -> JobRecord:
    async with asyncio.timeout(20):
        while True:
            record = await _record(jobs, world, job_id)
            if record.status == status:
                return record
            await asyncio.sleep(0.02)


class Running:
    def __init__(self, worker: Worker) -> None:
        self.stop_event = asyncio.Event()
        self.task = asyncio.create_task(worker.run(self.stop_event))

    async def stop(self) -> None:
        self.stop_event.set()
        await asyncio.wait_for(self.task, 20)


def _worker(db: Database, handlers: dict[str, object], name: str, **cfg: object) -> Worker:
    config = WorkerConfig(
        worker_id=name,
        **{"poll_interval": 0.05, "backoff_base": 0.05, "backoff_cap": 0.1, **cfg},  # type: ignore[arg-type]
    )
    return Worker(PgJobs(db), handlers, config)  # type: ignore[arg-type]


async def test_job_runs_to_done_with_progress(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND, {"n": 1}))

    async def handler(ctx: JobContext) -> None:
        await ctx.progress(1, 2)

    run = Running(_worker(app_db, {KIND: handler}, "w-done"))
    try:
        record = await _wait_status(jobs, world, job_id, "done")
    finally:
        await run.stop()
    assert record.progress == {"done": 1, "total": 2}
    assert record.attempts == 1
    assert record.finished_at is not None


async def test_failures_retry_with_backoff_then_fail_for_good(
    app_db: Database, world: World
) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    calls = 0

    async def handler(ctx: JobContext) -> None:
        nonlocal calls
        calls += 1
        raise ValueError(f"boom {calls}")

    run = Running(_worker(app_db, {KIND: handler}, "w-retry", max_attempts=3))
    try:
        record = await _wait_status(jobs, world, job_id, "failed")
    finally:
        await run.stop()
    assert calls == 3
    assert record.attempts == 3
    assert record.last_error == "ValueError: boom 3"


async def test_retry_succeeds_on_second_attempt(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    seen: list[int] = []

    async def handler(ctx: JobContext) -> None:
        seen.append(len(seen))
        if len(seen) == 1:
            raise RuntimeError("transient")

    run = Running(_worker(app_db, {KIND: handler}, "w-flaky"))
    try:
        record = await _wait_status(jobs, world, job_id, "done")
    finally:
        await run.stop()
    assert record.attempts == 2
    assert record.last_error == "RuntimeError: transient"


async def test_permanent_error_fails_on_first_attempt(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))

    async def handler(ctx: JobContext) -> None:
        raise PermanentJobError("cannot ever work")

    run = Running(_worker(app_db, {KIND: handler}, "w-perm"))
    try:
        record = await _wait_status(jobs, world, job_id, "failed")
    finally:
        await run.stop()
    assert record.attempts == 1
    assert record.last_error == "cannot ever work"


async def test_future_jobs_wait_for_run_at(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND, delay=timedelta(seconds=1)))
    ran = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        ran.set()

    run = Running(_worker(app_db, {KIND: handler}, "w-delay"))
    try:
        assert (await _record(jobs, world, job_id)).status == "queued"
        await asyncio.wait_for(ran.wait(), 20)
        await _wait_status(jobs, world, job_id, "done")
    finally:
        await run.stop()


async def test_expired_lease_is_reclaimed_after_a_crash(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    claimed = {c.id for c in await jobs.claim("w-crashed", 1000, timedelta(milliseconds=200))}
    assert job_id in claimed

    async def handler(ctx: JobContext) -> None:
        return None

    run = Running(_worker(app_db, {KIND: handler}, "w-rescue"))
    try:
        record = await _wait_status(jobs, world, job_id, "done")
    finally:
        await run.stop()
    assert record.attempts == 2


async def test_lease_is_renewed_so_a_slow_job_is_not_stolen(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    release = asyncio.Event()
    started = asyncio.Event()
    runs = 0

    async def slow(ctx: JobContext) -> None:
        nonlocal runs
        runs += 1
        started.set()
        await release.wait()

    async def other(ctx: JobContext) -> None:
        nonlocal runs
        runs += 1

    first = Running(_worker(app_db, {KIND: slow}, "w-slow", lease=timedelta(seconds=1)))
    try:
        await asyncio.wait_for(started.wait(), 20)
        second = Running(_worker(app_db, {KIND: other}, "w-thief", lease=timedelta(seconds=1)))
        try:
            await asyncio.sleep(2.5)
            assert (await _record(jobs, world, job_id)).status == "running"
            assert runs == 1
        finally:
            await second.stop()
        release.set()
        await _wait_status(jobs, world, job_id, "done")
    finally:
        await first.stop()


async def test_two_workers_never_run_a_job_twice(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    ids = [await jobs.enqueue(world.tenant, NewJob(KIND, {"i": i})) for i in range(20)]
    counts: dict[uuid.UUID, int] = {}

    async def handler(ctx: JobContext) -> None:
        counts[ctx.job.id] = counts.get(ctx.job.id, 0) + 1
        await asyncio.sleep(0.01)

    runs = [
        Running(_worker(app_db, {KIND: handler}, f"w-par-{n}", concurrency=3)) for n in range(2)
    ]
    try:
        for job_id in ids:
            await _wait_status(jobs, world, job_id, "done")
    finally:
        for run in runs:
            await run.stop()
    assert {i: counts[i] for i in ids} == dict.fromkeys(ids, 1)


async def test_graceful_shutdown_finishes_the_running_job(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(ctx: JobContext) -> None:
        started.set()
        await release.wait()

    run = Running(_worker(app_db, {KIND: handler}, "w-stop", shutdown_timeout=20))
    await asyncio.wait_for(started.wait(), 20)
    run.stop_event.set()
    await asyncio.sleep(0.2)
    assert not run.task.done()
    release.set()
    await asyncio.wait_for(run.task, 20)
    assert (await _record(jobs, world, job_id)).status == "done"


async def test_shutdown_timeout_leaves_the_job_reclaimable(app_db: Database, world: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(world.tenant, NewJob(KIND))
    started = asyncio.Event()

    async def hang(ctx: JobContext) -> None:
        started.set()
        await asyncio.Event().wait()

    run = Running(
        _worker(app_db, {KIND: hang}, "w-hang", shutdown_timeout=0.1, lease=timedelta(seconds=1))
    )
    await asyncio.wait_for(started.wait(), 20)
    await run.stop()
    assert (await _record(jobs, world, job_id)).status == "running"

    async def handler(ctx: JobContext) -> None:
        return None

    rescue = Running(_worker(app_db, {KIND: handler}, "w-after"))
    try:
        record = await _wait_status(jobs, world, job_id, "done")
    finally:
        await rescue.stop()
    assert record.attempts == 2
