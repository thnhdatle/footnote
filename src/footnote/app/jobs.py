# SPDX-License-Identifier: Apache-2.0
"""Job handlers for the queue worker: one per job kind.

`JobHandlers.handlers()` maps every kind except `webhook` to a handler taking a `JobRun`
(the worker's job context). The `webhook` handler is an adapter
(`adapters/webhooks`); the composition root merges both, so the worker knows every kind
in `JobKind`. Handlers read their arguments from the job row, raise
`PermanentJobError` for arguments or states a retry cannot fix, and let other errors
propagate so the worker retries with backoff.

`tenant_delete` runs over the admin pool (`TenantAdmin`): footnote_app has no privileges
on `tenants` and `api_keys`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from footnote.app.gc import GcService
from footnote.app.ingest import IngestFailedError, IngestService, IngestTarget
from footnote.app.reembed import ReembedService
from footnote.app.reindex import ReindexService
from footnote.domain.documents import CollectionId, DocumentId, VersionId
from footnote.domain.errors import NotFoundError, ValidationError
from footnote.domain.jobs import JobKind, PermanentJobError
from footnote.domain.ports import CollectionRepository, JobQueue, JobRun, TenantAdmin
from footnote.domain.settings import Language
from footnote.domain.tenancy import TenantStatus

type Handler = Callable[[JobRun], Awaitable[None]]


def _uuid(args: Mapping[str, Any], key: str) -> UUID:
    try:
        return UUID(str(args[key]))
    except (KeyError, ValueError) as exc:
        msg = f"job argument {key!r} missing or not a uuid"
        raise PermanentJobError(msg) from exc


@dataclass(frozen=True, slots=True)
class JobServices:
    """Use cases the handlers delegate to; `tenants` is the admin-pool tenant repository."""

    jobs: JobQueue
    collections: CollectionRepository
    ingest: IngestService
    reindex: ReindexService
    reembed: ReembedService
    gc: GcService
    tenants: TenantAdmin
    max_attempts: int = 5


class JobHandlers:
    """Handlers for every job kind but `webhook`."""

    def __init__(self, services: JobServices) -> None:
        self._s = services

    def handlers(self) -> dict[str, Handler]:
        """Kind -> handler (register with the worker together with the webhook handler)."""
        return {
            JobKind.INGEST.value: self.ingest,
            JobKind.REINDEX.value: self.reindex,
            JobKind.FTS_RECONFIG.value: self.fts_reconfig,
            JobKind.REEMBED.value: self.reembed,
            JobKind.GC.value: self.gc,
            JobKind.COLLECTION_DELETE.value: self.collection_delete,
            JobKind.TENANT_DELETE.value: self.tenant_delete,
        }

    async def _args(self, run: JobRun) -> tuple[dict[str, Any], int]:
        record = await self._s.jobs.get(run.job.tenant_id, run.job.id)
        return dict(record.args), record.attempts

    async def ingest(self, run: JobRun) -> None:
        """Index one uploaded (or reindexed) version."""
        args, attempts = await self._args(run)
        target = IngestTarget(
            CollectionId(_uuid(args, "collection_id")),
            DocumentId(_uuid(args, "document_id")),
            VersionId(_uuid(args, "version_id")),
        )
        try:
            await self._s.ingest.ingest(
                run.job.tenant_id, target, last_attempt=attempts >= self._s.max_attempts
            )
        except IngestFailedError as exc:
            raise PermanentJobError(f"{exc.code}: {exc.message}") from exc

    async def reindex(self, run: JobRun) -> None:
        """Re-chunk a collection's documents after a chunking change."""
        args, _ = await self._args(run)
        collection_id = CollectionId(_uuid(args, "collection_id"))
        await self._s.reindex.reindex(run.job.tenant_id, collection_id, run.progress)

    async def fts_reconfig(self, run: JobRun) -> None:
        """Switch a collection's FTS language (one transaction)."""
        args, _ = await self._args(run)
        collection_id = CollectionId(_uuid(args, "collection_id"))
        try:
            language = Language(str(args["language"]))
        except (KeyError, ValueError) as exc:
            raise PermanentJobError("job argument 'language' missing or unsupported") from exc
        await self._s.reindex.fts_reconfig(run.job.tenant_id, collection_id, language)

    async def reembed(self, run: JobRun) -> None:
        """Backfill a collection's target space and switch."""
        args, _ = await self._args(run)
        collection_id = CollectionId(_uuid(args, "collection_id"))
        try:
            target = int(args["target_space_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PermanentJobError("job argument 'target_space_id' missing") from exc
        await self._s.reembed.run(run.job.tenant_id, collection_id, target, run.progress)

    async def gc(self, run: JobRun) -> None:
        """Post-swap / post-switch garbage collection."""
        args, _ = await self._args(run)
        try:
            await self._s.gc.run(run.job.tenant_id, args)
        except (ValidationError, ValueError) as exc:
            raise PermanentJobError(str(exc)) from exc

    async def collection_delete(self, run: JobRun) -> None:
        """Remove a `deleting` collection's rows."""
        args, _ = await self._args(run)
        collection_id = CollectionId(_uuid(args, "collection_id"))
        tenant = run.job.tenant_id
        try:
            collection = await self._s.collections.get(tenant, collection_id, include_deleting=True)
        except NotFoundError:
            return
        if collection.status != "deleting":
            raise PermanentJobError("collection is not being deleted")
        await self._s.collections.delete_rows(tenant, collection_id)

    async def tenant_delete(self, run: JobRun) -> None:
        """Remove every row of a `deleting` tenant, this job's row included (admin pool)."""
        tenant_id = run.job.tenant_id
        try:
            tenant = await self._s.tenants.get_tenant(tenant_id)
        except NotFoundError:
            return
        if tenant.status is not TenantStatus.DELETING:
            raise PermanentJobError("tenant is not being deleted")
        await self._s.tenants.delete_tenant_data(tenant_id)
