# SPDX-License-Identifier: Apache-2.0
"""Re-embedding on a space change: backfill, dual-write, atomic switch.

`change_space` sets the collection's target space and enqueues a `reembed` job (or, when
the requested space is the active one, cancels a running re-embed and schedules GC of
its partial vectors). The job embeds current-version chunks lacking a target vector in
batches; meanwhile ingestion writes both spaces (the swap verifies every needed space).
Queries read only the active space, so they never mix spaces. When nothing lacks a
target vector, the switch transaction makes the target active (exactly once) and
schedules GC of the old space's vectors after `gc_grace`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from footnote.domain.documents import ChunkId, CollectionId, JobId
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import ConflictError, NotFoundError, UpstreamError
from footnote.domain.jobs import JobKind, NewJob
from footnote.domain.ports import (
    CollectionRepository,
    Embedder,
    EmbedderFactory,
    IndexRepository,
    JobQueue,
    SpaceReader,
)
from footnote.domain.tenancy import TenantId

type Progress = Callable[[int, int], Awaitable[None]]


class ReembedStatus(StrEnum):
    """How a `reembed` run ended."""

    SWITCHED = "switched"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ReembedConfig:
    """`embedding.batch_size`, `ingest.gc_grace` and the switch retry bound."""

    batch_size: int = 64
    gc_grace: timedelta = timedelta(minutes=10)
    max_switch_attempts: int = 100


@dataclass(frozen=True, slots=True)
class ReembedOutcome:
    """Result of one run; `embedded` counts texts embedded by this run."""

    status: ReembedStatus
    embedded: int


async def _no_progress(done: int, total: int) -> None:
    """Default progress sink."""


def gc_space_job(collection_id: CollectionId, space_id: int, delay: timedelta) -> NewJob:
    """`gc` of one space's vectors in a collection (no-op once the space is in use again)."""
    args = {"collection_id": str(collection_id), "space_id": space_id}
    return NewJob(JobKind.GC.value, args, delay)


class ReembedService:
    """Starts, cancels and runs re-embeds."""

    def __init__(  # noqa: PLR0913 - the use case's ports
        self,
        collections: CollectionRepository,
        index: IndexRepository,
        spaces: SpaceReader,
        embedders: EmbedderFactory,
        jobs: JobQueue,
        *,
        config: ReembedConfig | None = None,
    ) -> None:
        self._collections = collections
        self._index = index
        self._spaces = spaces
        self._embedders = embedders
        self._jobs = jobs
        self._cfg = config or ReembedConfig()

    async def change_space(
        self, tenant: TenantId, collection_id: CollectionId, space_name: str
    ) -> JobId | None:
        """Point the collection at `space_name`: a `reembed` job id, or None for no job.

        Choosing the active space cancels a running re-embed (GC of its partial vectors);
        choosing the current target again changes nothing unless its `reembed` job is gone
        (the target was set but the enqueue failed), in which case the job is enqueued again.
        """
        space = await self._spaces.get(space_name)
        collection = await self._collections.get(tenant, collection_id)
        if space.id == collection.target_space_id:
            return await self._ensure_job(tenant, collection_id, space.id)
        previous = collection.target_space_id
        if space.id == collection.active_space_id:
            if previous is not None:
                await self._collections.set_target_space(tenant, collection_id, None)
                await self._jobs.enqueue(
                    tenant, gc_space_job(collection_id, previous, self._cfg.gc_grace)
                )
            return None
        await self._collections.set_target_space(tenant, collection_id, space.id)
        if previous is not None:
            await self._jobs.enqueue(
                tenant, gc_space_job(collection_id, previous, self._cfg.gc_grace)
            )
        return await self._enqueue_reembed(tenant, collection_id, space.id)

    async def _enqueue_reembed(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> JobId:
        args = {"collection_id": str(collection_id), "target_space_id": space_id}
        return await self._jobs.enqueue(tenant, NewJob(JobKind.REEMBED.value, args))

    async def _ensure_job(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> JobId | None:
        for job in await self._jobs.active_for_collection(tenant, collection_id):
            if job.kind == JobKind.REEMBED.value and job.args.get("target_space_id") == space_id:
                return None
        return await self._enqueue_reembed(tenant, collection_id, space_id)

    async def run(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        target_space_id: int,
        progress: Progress = _no_progress,
    ) -> ReembedOutcome:
        """Backfill target vectors, then switch; stops when the target changed meanwhile."""
        try:
            collection = await self._collections.get(tenant, collection_id)
        except NotFoundError:
            return ReembedOutcome(ReembedStatus.CANCELLED, 0)
        if collection.target_space_id != target_space_id:
            return ReembedOutcome(ReembedStatus.CANCELLED, 0)
        old_active = collection.active_space_id
        space = await self._spaces.get_by_id(target_space_id)
        embedder = self._embedders.for_space(space)
        gc = gc_space_job(collection_id, old_active, self._cfg.gc_grace)
        embedded = 0
        stalls = 0
        while await self._still_target(tenant, collection_id, target_space_id):
            batch = await self._index.reembed_batch(
                tenant, collection_id, target_space_id, self._cfg.batch_size
            )
            if batch:
                embedded += await self._backfill(
                    tenant, collection_id, space, embedder, batch, progress=progress
                )
                continue
            if await self._index.switch_space(tenant, collection_id, jobs=(gc,)):
                return ReembedOutcome(ReembedStatus.SWITCHED, embedded)
            # A swap added a chunk between the batch query and the switch: go round again.
            stalls += 1
            if stalls >= self._cfg.max_switch_attempts:
                msg = "re-embed did not converge; retrying"
                raise ConflictError(msg, code="reembed_incomplete")
        return ReembedOutcome(ReembedStatus.CANCELLED, embedded)

    async def _backfill(  # noqa: PLR0913 - one batch of the run's state
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        space: EmbeddingSpace,
        embedder: Embedder,
        batch: list[tuple[ChunkId, str]],
        *,
        progress: Progress,
    ) -> int:
        vectors = await embedder.embed_documents([text for _, text in batch])
        if len(vectors) != len(batch):
            msg = f"embedder returned {len(vectors)} vectors for {len(batch)} texts"
            raise UpstreamError(msg)
        ids = [chunk_id for chunk_id, _ in batch]
        await self._index.insert_vectors(
            tenant, collection_id, space, list(zip(ids, vectors, strict=True))
        )
        done, total = await self._index.reembed_counts(tenant, collection_id, space.id)
        await self._collections.set_reembed_progress(tenant, collection_id, done, total)
        await progress(done, total)
        return len(batch)

    async def _still_target(
        self, tenant: TenantId, collection_id: CollectionId, target_space_id: int
    ) -> bool:
        try:
            collection = await self._collections.get(tenant, collection_id)
        except NotFoundError:
            return False
        return collection.target_space_id == target_space_id
