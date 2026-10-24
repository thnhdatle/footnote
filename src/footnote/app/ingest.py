# SPDX-License-Identifier: Apache-2.0
"""Ingest use case: parse, chunk, diff, embed and swap one version."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any

from footnote.app.webhooks import WebhookNotifier
from footnote.domain.chunking import ChunkDiff, chunk_elements, diff_hashes
from footnote.domain.documents import (
    ChunkBatch,
    ChunkDraft,
    ChunkId,
    CollectionId,
    CollectionRecord,
    DocumentId,
    DocumentStatus,
    Element,
    Membership,
    SwapResult,
    VersionId,
    VersionRecord,
    VersionStatus,
)
from footnote.domain.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    UnavailableError,
    UpstreamError,
)
from footnote.domain.jobs import JobKind, NewJob
from footnote.domain.ports import (
    Clock,
    CollectionRepository,
    DocumentRepository,
    EmbedderFactory,
    IndexRepository,
    Parser,
    SpaceReader,
    TenantReader,
    TokenCounterFactory,
)
from footnote.domain.tenancy import TenantId
from footnote.domain.webhooks import WebhookEvent
from footnote.observability import tracer

VECTORS_MISSING = "vectors_missing"
_MAX_ERROR_CHARS = 500
_TRANSIENT = (UpstreamError, UnavailableError, ConflictError)

type StageObserver = Callable[[str, float], None]


class IngestStatus(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestConfig:
    gc_grace: timedelta = timedelta(minutes=10)
    retain_originals: bool = True
    batch_size: int = 64
    swap_attempts: int = 3


@dataclass(frozen=True, slots=True)
class IngestDeps:
    collections: CollectionRepository
    documents: DocumentRepository
    index: IndexRepository
    spaces: SpaceReader
    parser: Parser
    embedders: EmbedderFactory
    tokenizers: TokenCounterFactory
    webhooks: WebhookNotifier
    clock: Clock
    tenants: TenantReader | None = None


@dataclass(frozen=True, slots=True)
class IngestTarget:
    collection_id: CollectionId
    document_id: DocumentId
    version_id: VersionId


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    status: IngestStatus
    chunks: int = 0
    reused: int = 0
    new: int = 0
    embedded: int = 0
    error: dict[str, Any] | None = None


class IngestFailedError(DomainError):
    code = "ingest_failed"


def error_body(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, DomainError):
        return {"code": exc.code, "message": exc.message[:_MAX_ERROR_CHARS]}
    return {"code": "internal", "message": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]}


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT) or not isinstance(exc, DomainError)


def memberships_for(
    version_id: VersionId, drafts: Sequence[ChunkDraft], ids: dict[str, ChunkId]
) -> list[Membership]:
    return [
        Membership(version_id, d.ordinal, ids[d.content_hash], d.page_from, d.page_to)
        for d in drafts
    ]


def page_count(elements: Sequence[Element]) -> int | None:
    pages = [p for e in elements for p in (e.page, e.page_to) if p is not None]
    return max(pages) if pages else None


@dataclass(slots=True)
class _Run:
    tenant: TenantId
    target: IngestTarget
    version: VersionRecord
    collection: CollectionRecord
    drafts: list[ChunkDraft]
    ids: dict[str, ChunkId]
    diff: ChunkDiff
    embedded: int = 0


class IngestService:
    """Runs `ingest` jobs."""

    def __init__(
        self,
        deps: IngestDeps,
        config: IngestConfig | None = None,
        on_stage: StageObserver | None = None,
    ) -> None:
        self._d = deps
        self._cfg = config or IngestConfig()
        self._on_stage = on_stage

    async def ingest(
        self, tenant: TenantId, target: IngestTarget, *, last_attempt: bool = False
    ) -> IngestOutcome:
        """Indexes one version."""
        with tracer().start_as_current_span("ingest document"):
            try:
                version = await self._d.documents.version(tenant, target.version_id)
                collection = await self._d.collections.get(tenant, target.collection_id)
            except NotFoundError:
                return IngestOutcome(IngestStatus.SKIPPED)
            if version.status is not VersionStatus.PROCESSING:
                await self._drop_original(tenant, version)
                return IngestOutcome(IngestStatus.SKIPPED)
            try:
                return await self._index(tenant, target, version, collection)
            except Exception as exc:
                if is_transient(exc) and not last_attempt:
                    raise
                error = error_body(exc)
                try:
                    await self._fail(tenant, version, collection, error)
                except NotFoundError:
                    return IngestOutcome(IngestStatus.SKIPPED)
                raise IngestFailedError(error["message"], code=error["code"]) from exc

    async def _index(
        self,
        tenant: TenantId,
        target: IngestTarget,
        version: VersionRecord,
        collection: CollectionRecord,
    ) -> IngestOutcome:
        elements = await self._elements(tenant, version)
        start = self._d.clock.monotonic()
        with tracer().start_as_current_span("chunk"):
            settings = collection.settings.chunking
            counter = self._d.tokenizers.for_tokenizer(settings.tokenizer)
            drafts = chunk_elements(elements, version.indexed_title, settings, counter)
        self._stage("chunk", start)
        await self._check_quota(tenant, target.document_id, drafts)
        await self._d.documents.set_status(
            tenant, target.document_id, DocumentStatus.EMBEDDING, version_id=version.id
        )
        run = await self._insert(tenant, target, version, collection, drafts)
        start = self._d.clock.monotonic()
        with tracer().start_as_current_span("embed"):
            run.embedded += await self._embed_missing(run)
        self._stage("embed", start)
        result = await self._swap(run)
        if result.current:
            await self._drop_original(tenant, version)
        return IngestOutcome(
            IngestStatus.CURRENT if result.current else IngestStatus.SUPERSEDED,
            chunks=len(run.ids),
            reused=len(run.diff.reused),
            new=len(run.diff.new),
            embedded=run.embedded,
        )

    async def _elements(self, tenant: TenantId, version: VersionRecord) -> list[Element]:
        try:
            return await self._d.documents.elements(tenant, version.id)
        except NotFoundError:
            pass
        await self._d.documents.set_status(
            tenant, version.document_id, DocumentStatus.PARSING, version_id=version.id
        )
        start = self._d.clock.monotonic()
        with tracer().start_as_current_span("parse"):
            data = await self._d.documents.source(tenant, version.id)
            elements = (await self._d.parser.parse(version.media_type, data)).elements
        self._stage("parse", start)
        await self._d.documents.store_elements(
            tenant, version.id, elements, pages=page_count(elements)
        )
        return elements

    async def _check_quota(
        self, tenant: TenantId, document_id: DocumentId, drafts: Sequence[ChunkDraft]
    ) -> None:
        if self._d.tenants is None:
            return
        quotas = (await self._d.tenants.get_tenant(tenant)).quotas
        if not quotas.max_chunks:
            return
        total = await self._d.collections.total_chunks(tenant)
        current = await self._d.index.current_chunk_count(tenant, document_id)
        adding = len({d.content_hash for d in drafts})
        quotas.check("max_chunks", total - current, adding)

    async def _insert(
        self,
        tenant: TenantId,
        target: IngestTarget,
        version: VersionRecord,
        collection: CollectionRecord,
        drafts: list[ChunkDraft],
    ) -> _Run:
        hashes = sorted({d.content_hash for d in drafts})
        existing = await self._d.index.existing_chunks(tenant, target.document_id, hashes)
        diff = diff_hashes(existing, drafts)
        fresh = [d for d in drafts if d.content_hash in diff.new]
        ids = dict(existing)
        if fresh:
            batch = ChunkBatch(
                collection.id,
                target.document_id,
                version.indexed_title,
                collection.settings.language,
                fresh,
            )
            ids.update(await self._d.index.insert_chunks(tenant, batch))
        ids = {h: ids[h] for h in hashes}
        return _Run(tenant, target, version, collection, drafts, ids, diff)

    async def _embed_missing(self, run: _Run) -> int:
        texts = {run.ids[d.content_hash]: d.embedded_text for d in run.drafts}
        missing = await self._d.index.missing_vectors(run.tenant, run.collection.id, list(texts))
        embedded = 0
        for space_id in sorted(missing):
            space = await self._d.spaces.get_by_id(space_id)
            embedder = self._d.embedders.for_space(space)
            chunk_ids = missing[space_id]
            for start in range(0, len(chunk_ids), self._cfg.batch_size):
                batch = chunk_ids[start : start + self._cfg.batch_size]
                vectors = await embedder.embed_documents([texts[c] for c in batch])
                if len(vectors) != len(batch):
                    msg = f"embedder returned {len(vectors)} vectors for {len(batch)} texts"
                    raise UpstreamError(msg)
                await self._d.index.insert_vectors(
                    run.tenant, run.collection.id, space, list(zip(batch, vectors, strict=True))
                )
                embedded += len(batch)
        return embedded

    async def _swap(self, run: _Run) -> SwapResult:
        doc = run.target.document_id
        gc = NewJob(JobKind.GC.value, {"document_id": str(doc)}, self._cfg.gc_grace)
        ready = self._d.webhooks.job_for(run.collection, WebhookEvent.DOCUMENT_READY, run.version)
        memberships = memberships_for(run.version.id, run.drafts, run.ids)
        start = self._d.clock.monotonic()
        attempt = 1
        with tracer().start_as_current_span("swap"):
            while True:
                try:
                    result = await self._d.index.swap(
                        run.tenant,
                        doc,
                        run.version.id,
                        memberships,
                        jobs=(gc,),
                        on_current=() if ready is None else (ready,),
                    )
                    break
                except ConflictError as exc:
                    if exc.code != VECTORS_MISSING or attempt >= self._cfg.swap_attempts:
                        raise
                    attempt += 1
                    run.embedded += await self._embed_missing(run)
        self._stage("swap", start)
        return result

    async def _drop_original(self, tenant: TenantId, version: VersionRecord) -> None:
        if not self._cfg.retain_originals and version.status in (
            VersionStatus.PROCESSING,
            VersionStatus.READY,
        ):
            await self._d.documents.delete_original(tenant, version.id)

    async def _fail(
        self,
        tenant: TenantId,
        version: VersionRecord,
        collection: CollectionRecord,
        error: dict[str, Any],
    ) -> None:
        await self._d.documents.fail_version(tenant, version.id, error)
        await self._d.webhooks.notify(
            tenant, collection, WebhookEvent.DOCUMENT_FAILED, version, error=error
        )

    def _stage(self, stage: str, start: float) -> None:
        if self._on_stage is not None:
            self._on_stage(stage, self._d.clock.monotonic() - start)
