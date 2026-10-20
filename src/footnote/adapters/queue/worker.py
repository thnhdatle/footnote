# SPDX-License-Identifier: Apache-2.0
"""Worker loop: claims jobs across tenants and runs one handler per job kind.

A claimed job runs as its own task next to a lease renewer. When the renewer finds the
lease lost (reclaimed after expiry) it cancels the handler; the repository's
`lease_owner` guard already stops a stale worker from overwriting the new owner.
Failures requeue with jittered backoff until `max_attempts`, then fail for good;
`PermanentJobError` fails immediately. On stop the loop claims nothing more, waits up to
`shutdown_timeout` for running jobs and cancels the rest (their leases expire and another
worker reclaims them). `on_outcome(kind, status)` reports every recorded outcome
(`succeeded`, `retried`, `failed`) for `footnote_jobs_total`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from footnote.domain.documents import JobId
from footnote.domain.jobs import ClaimedJob, JobRecord, PermanentJobError
from footnote.domain.tenancy import TenantId
from footnote.domain.webhooks import backoff_seconds

_MAX_ERROR_CHARS = 2000


class JobStore(Protocol):
    """What the worker needs from the job repository (`PgJobs` implements it)."""

    async def claim(self, worker: str, n: int, lease: timedelta) -> list[ClaimedJob]:
        """Claim up to `n` due jobs."""

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        """One job (for its attempt count)."""

    async def extend_lease(
        self, tenant: TenantId, job_id: JobId, worker: str, lease: timedelta
    ) -> bool:
        """Extend the lease; False when the worker no longer owns the job."""

    async def set_progress(
        self, tenant: TenantId, job_id: JobId, worker: str, done: int, total: int
    ) -> bool:
        """Record progress."""

    async def complete(self, tenant: TenantId, job_id: JobId, worker: str) -> bool:
        """Mark done."""

    async def fail(
        self,
        tenant: TenantId,
        job_id: JobId,
        worker: str,
        error: str,
        retry_in: timedelta | None,
    ) -> bool:
        """Requeue after `retry_in` or fail for good when None."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """`ingest.*` settings of the worker (workers, job_lease, job_max_attempts, poll_interval)."""

    worker_id: str
    concurrency: int = 4
    lease: timedelta = timedelta(seconds=60)
    max_attempts: int = 5
    poll_interval: float = 1.0
    backoff_base: float = 2.0
    backoff_cap: float = 300.0
    shutdown_timeout: float = 30.0

    @property
    def renew_interval(self) -> float:
        """Seconds between lease extensions (a third of the lease)."""
        return self.lease.total_seconds() / 3


@dataclass(frozen=True, slots=True)
class JobContext:
    """What a handler receives: the claimed job and a progress reporter."""

    job: ClaimedJob
    store: JobStore
    worker: str

    async def progress(self, done: int, total: int) -> None:
        """Record `{done, total}` progress for the job."""
        await self.store.set_progress(self.job.tenant_id, self.job.id, self.worker, done, total)


type Handler = Callable[[JobContext], Awaitable[None]]
type OutcomeObserver = Callable[[str, str], None]


def _ignore_outcome(kind: str, status: str) -> None:
    """Default `on_outcome`: no metrics."""


async def _invoke(handler: Handler, ctx: JobContext) -> None:
    await handler(ctx)


class Worker:
    """Runs `config.concurrency` jobs at a time until `run` is told to stop."""

    def __init__(  # noqa: PLR0913 - optional rand, log and metrics observer
        self,
        store: JobStore,
        handlers: Mapping[str, Handler],
        config: WorkerConfig,
        *,
        rand: Callable[[], float] = random.random,
        log: Any = None,
        on_outcome: OutcomeObserver = _ignore_outcome,
    ) -> None:
        self._store = store
        self._handlers = dict(handlers)
        self._cfg = config
        self._rand = rand
        self._log = log if log is not None else structlog.get_logger("footnote.queue")
        self._running: set[asyncio.Task[None]] = set()
        self._on_outcome = on_outcome

    async def run(self, stop: asyncio.Event) -> None:
        """Claim and process jobs until `stop` is set, then drain (bounded)."""
        stop_wait = asyncio.create_task(stop.wait())
        try:
            while not stop.is_set():
                claimed = await self._claim_free()
                for job in claimed:
                    task = asyncio.create_task(self._run_job(job))
                    self._running.add(task)
                    task.add_done_callback(self._running.discard)
                saturated = len(self._running) >= self._cfg.concurrency
                timeout = None if saturated else self._cfg.poll_interval
                await asyncio.wait(
                    {stop_wait, *self._running},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            stop_wait.cancel()
            await self._drain()

    async def _claim_free(self) -> Sequence[ClaimedJob]:
        free = self._cfg.concurrency - len(self._running)
        if free <= 0:
            return ()
        try:
            return await self._store.claim(self._cfg.worker_id, free, self._cfg.lease)
        except Exception as exc:  # DB blip: log and retry on the next poll
            self._log.error("claim failed", component="queue", error=str(exc))
            return ()

    async def _drain(self) -> None:
        pending = set(self._running)
        if not pending:
            return
        _, still = await asyncio.wait(pending, timeout=self._cfg.shutdown_timeout)
        for task in still:
            task.cancel()
        if still:
            self._log.warning("cancelled running jobs", component="queue", count=len(still))
            await asyncio.gather(*still, return_exceptions=True)

    async def _run_job(self, job: ClaimedJob) -> None:
        handler = self._handlers.get(job.kind)
        if handler is None:
            await self._fail(job, f"no handler for job kind {job.kind!r}", permanent=True)
            return
        ctx = JobContext(job, self._store, self._cfg.worker_id)
        work = asyncio.create_task(_invoke(handler, ctx))
        renewer = asyncio.create_task(self._renew(job))
        try:
            await asyncio.wait({work, renewer}, return_when=asyncio.FIRST_COMPLETED)
            if not work.done():
                self._log.warning("lease lost", component="queue", job_id=str(job.id))
                return
            await self._finish(job, work)
        finally:
            for task in (work, renewer):
                task.cancel()
            await asyncio.gather(work, renewer, return_exceptions=True)

    async def _finish(self, job: ClaimedJob, work: asyncio.Task[None]) -> None:
        exc = work.exception() if not work.cancelled() else asyncio.CancelledError()
        if exc is None:
            if await self._guarded(
                self._store.complete(job.tenant_id, job.id, self._cfg.worker_id), job
            ):
                self._on_outcome(job.kind, "succeeded")
        elif isinstance(exc, PermanentJobError):
            await self._fail(job, str(exc), permanent=True)
        elif isinstance(exc, Exception):
            await self._fail(job, f"{type(exc).__name__}: {exc}", permanent=False)
        else:
            await self._fail(job, "handler was cancelled", permanent=False)

    async def _renew(self, job: ClaimedJob) -> None:
        """Extend the lease until it is lost; returns only on loss."""
        while True:
            await asyncio.sleep(self._cfg.renew_interval)
            try:
                owned = await self._store.extend_lease(
                    job.tenant_id, job.id, self._cfg.worker_id, self._cfg.lease
                )
            except Exception as exc:  # transient DB error: try again before the lease ends
                self._log.warning(
                    "lease renewal failed", component="queue", job_id=str(job.id), error=str(exc)
                )
                continue
            if not owned:
                return

    async def _fail(self, job: ClaimedJob, error: str, *, permanent: bool) -> None:
        error = error[:_MAX_ERROR_CHARS]
        try:
            retry_in = None if permanent else await self._retry_in(job)
            recorded = await self._guarded(
                self._store.fail(job.tenant_id, job.id, self._cfg.worker_id, error, retry_in),
                job,
            )
        except Exception as exc:  # the lease will expire and the job is reclaimed
            self._log.error(
                "recording failure failed", component="queue", job_id=str(job.id), error=str(exc)
            )
            return
        if recorded:
            self._on_outcome(job.kind, "failed" if retry_in is None else "retried")
        self._log.error(
            "job failed",
            component="queue",
            job_id=str(job.id),
            tenant=str(job.tenant_id),
            error=error,
        )

    async def _retry_in(self, job: ClaimedJob) -> timedelta | None:
        record = await self._store.get(job.tenant_id, job.id)
        if record.attempts >= self._cfg.max_attempts:
            return None
        seconds = backoff_seconds(
            max(record.attempts, 1), self._cfg.backoff_base, self._cfg.backoff_cap, self._rand()
        )
        return timedelta(seconds=seconds)

    async def _guarded(self, call: Awaitable[bool], job: ClaimedJob) -> bool:
        """Run a state update; True when it was recorded (the lease was still ours)."""
        try:
            owned = await call
        except Exception as exc:  # the lease will expire and the job is reclaimed
            self._log.error(
                "recording job state failed", component="queue", job_id=str(job.id), error=str(exc)
            )
            return False
        if not owned:
            self._log.warning("lease lost", component="queue", job_id=str(job.id))
        return owned


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_jobrun(ctx: JobContext) -> ports.JobRun:
        return ctx
