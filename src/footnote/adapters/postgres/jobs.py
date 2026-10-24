# SPDX-License-Identifier: Apache-2.0
"""Job rows of the queue."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from psycopg import sql

from footnote.adapters.postgres.session import (
    Conn,
    Database,
    IdFactory,
    execute,
    fetch_all,
    fetch_one,
    jsonb,
    new_id,
    require,
)
from footnote.domain.documents import CollectionId, JobId
from footnote.domain.jobs import ClaimedJob, JobRecord, NewJob
from footnote.domain.tenancy import TenantId


async def insert_job(conn: Conn, tenant: TenantId, job: NewJob, job_id: JobId) -> None:
    await conn.execute(
        "INSERT INTO footnote.jobs (tenant_id, id, kind, args, run_at) "
        "VALUES (%(tenant)s, %(id)s, %(kind)s, %(args)s, now() + %(delay)s)",
        {
            "tenant": tenant,
            "id": job_id,
            "kind": job.kind,
            "args": jsonb(dict(job.args)),
            "delay": job.delay,
        },
    )


def _record(row: dict[str, Any]) -> JobRecord:
    return JobRecord(
        id=JobId(row["id"]),
        tenant_id=TenantId(row["tenant_id"]),
        kind=row["kind"],
        args=row["args"],
        status=row["status"],
        attempts=int(row["attempts"]),
        last_error=row["last_error"],
        progress=row["progress"],
        run_at=row["run_at"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


_COLS = sql.SQL(
    "SELECT j.id, j.tenant_id, j.kind, j.args, j.status, j.attempts, j.last_error, j.progress, "
    "j.run_at, j.created_at, j.started_at, j.finished_at, j.updated_at FROM footnote.jobs j "
)


class PgJobs:
    """Job repository."""

    def __init__(self, db: Database, id_factory: IdFactory = new_id) -> None:
        self._db = db
        self._new_id = id_factory

    def new_job_id(self) -> JobId:
        return JobId(self._new_id())

    async def enqueue(self, tenant: TenantId, job: NewJob) -> JobId:
        """Inserts a queued job."""
        job_id = self.new_job_id()
        async with self._db.tenant_tx(tenant) as conn:
            await insert_job(conn, tenant, job, job_id)
        return job_id

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        query = sql.SQL("{} WHERE {} AND j.id = %(id)s").format(_COLS, self._db.tenant_filter("j"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "id": job_id})
        return _record(require(row, "job not found"))

    async def active_for_collection(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> list[JobRecord]:
        query = sql.SQL(
            "{} WHERE {} AND j.status IN ('queued', 'running') "
            "AND j.args ->> 'collection_id' = %(cid)s ORDER BY j.created_at, j.id"
        ).format(_COLS, self._db.tenant_filter("j"))
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(conn, query, {"tenant": tenant, "cid": str(collection_id)})
        return [_record(r) for r in rows]

    async def claim(self, worker: str, n: int, lease: timedelta) -> list[ClaimedJob]:
        """Claims due jobs across tenants."""
        async with self._db.global_tx() as conn:
            rows = await fetch_all(
                conn,
                "SELECT job_id, tenant_id, kind FROM footnote.footnote_claim_jobs("
                "%(worker)s, %(n)s, %(lease)s)",
                {"worker": worker, "n": n, "lease": lease},
            )
        return [ClaimedJob(JobId(r["job_id"]), TenantId(r["tenant_id"]), r["kind"]) for r in rows]

    async def _owned_update(
        self, tenant: TenantId, assignments: sql.Composable, params: dict[str, Any]
    ) -> bool:
        query = sql.SQL(
            "UPDATE footnote.jobs j SET {}, updated_at = now() WHERE {} AND j.id = %(id)s "
            "AND j.status = 'running' AND j.lease_owner = %(worker)s"
        ).format(assignments, self._db.tenant_filter("j"))
        async with self._db.tenant_tx(tenant) as conn:
            return await execute(conn, query, {"tenant": tenant, **params}) == 1

    async def extend_lease(
        self, tenant: TenantId, job_id: JobId, worker: str, lease: timedelta
    ) -> bool:
        return await self._owned_update(
            tenant,
            sql.SQL("lease_until = now() + %(lease)s"),
            {"id": job_id, "worker": worker, "lease": lease},
        )

    async def set_progress(
        self, tenant: TenantId, job_id: JobId, worker: str, done: int, total: int
    ) -> bool:
        return await self._owned_update(
            tenant,
            sql.SQL(
                "progress = jsonb_build_object('done', %(done)s::bigint, "
                "'total', %(total)s::bigint)"
            ),
            {"id": job_id, "worker": worker, "done": done, "total": total},
        )

    async def complete(self, tenant: TenantId, job_id: JobId, worker: str) -> bool:
        return await self._owned_update(
            tenant,
            sql.SQL("status = 'done', finished_at = now(), lease_owner = NULL, lease_until = NULL"),
            {"id": job_id, "worker": worker},
        )

    async def fail(
        self,
        tenant: TenantId,
        job_id: JobId,
        worker: str,
        error: str,
        retry_in: timedelta | None,
    ) -> bool:
        return await self._owned_update(
            tenant,
            sql.SQL(
                "last_error = %(error)s, lease_owner = NULL, lease_until = NULL, "
                "status = CASE WHEN %(retry)s::interval IS NULL THEN 'failed' ELSE 'queued' END, "
                "run_at = COALESCE(now() + %(retry)s::interval, j.run_at), "
                "finished_at = CASE WHEN %(retry)s::interval IS NULL THEN now() END"
            ),
            {"id": job_id, "worker": worker, "error": error, "retry": retry_in},
        )


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_jobqueue(db: Database) -> ports.JobQueue:
        return PgJobs(db)
