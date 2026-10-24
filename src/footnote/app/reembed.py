# SPDX-License-Identifier: Apache-2.0
"""Re-embedding after an embedding space change."""

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
    SWITCHED = "switched"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ReembedConfig:
    batch_size: int = 64
    gc_grace: timedelta = timedelta(minutes=10)
    max_switch_attempts: int = 100


@dataclass(frozen=True, slots=True)
class ReembedOutcome:
    status: ReembedStatus
    embedded: int


async def _no_progress(done: int, total: int) -> None: ...


def gc_space_job(collection_id: CollectionId, space_id: int, delay: timedelta) -> NewJob:
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
        """Points the collection at another space and queues the re-embed."""
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
        """Backfills the target space and then switches to it."""
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
