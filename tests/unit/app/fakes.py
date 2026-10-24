# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from footnote.domain.answer import Usage
from footnote.domain.documents import (
    ChunkBatch,
    ChunkId,
    CollectionId,
    CollectionRecord,
    DocumentId,
    DocumentRecord,
    DocumentStatus,
    Element,
    ElementKind,
    GcStats,
    JobId,
    Membership,
    NewCollection,
    NewUpload,
    PageImage,
    ParsedDocument,
    SwapResult,
    UploadResult,
    VersionId,
    VersionRecord,
    VersionStatus,
)
from footnote.domain.embedding import EmbeddingSpace, cosine_similarity, l2_normalise
from footnote.domain.errors import ConflictError, NotFoundError, ValidationError
from footnote.domain.jobs import ClaimedJob, JobRecord, NewJob
from footnote.domain.ports import (
    ChatDelta,
    ChatMessage,
    ChatResult,
    CollectionView,
    Embedder,
    LexicalQuery,
    SearchFilters,
    SearchSnapshot,
    TokenCounter,
    VectorQuery,
)
from footnote.domain.retrieval import Hit, LegHit
from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.tenancy import Quotas, Tenant, TenantId

DIMS = 8
_WORD = re.compile(r"\w+")


def words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


class FakeClock:
    def __init__(self, step: float = 0.5) -> None:
        self.step = step
        self._t = 0.0

    def now(self) -> float:
        return 1_700_000_000.0 + self._t

    def monotonic(self) -> float:
        self._t += self.step
        return self._t


def hash_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(b - 127.5) / 127.5 for b in digest[:DIMS]]


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [hash_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return hash_vector(text)


class FakeEmbedderFactory:
    def __init__(self, embedder: FakeEmbedder | None = None) -> None:
        self.embedder = embedder or FakeEmbedder()
        self.spaces: list[EmbeddingSpace] = []

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        self.spaces.append(space)
        return self.embedder


class FakeReranker:
    model_key = "fake-reranker"

    def __init__(
        self,
        calibrated_threshold: float | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.calibrated_threshold = calibrated_threshold
        self.events = events if events is not None else []
        self.calls: list[tuple[str, list[str]]] = []
        self.wrong_count = False

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.events.append("rerank")
        self.calls.append((query, list(passages)))
        q = words(query)
        logits = [float(len(q & words(p)) - 1) for p in passages]
        return logits[:-1] if self.wrong_count else logits


class FakeLLM:
    def __init__(  # noqa: PLR0913 - scripted fake with many knobs
        self,
        text: str = "",
        *,
        configured: bool = True,
        chunk_size: int = 4,
        usage: Usage | None = None,
        fail_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.configured = configured
        self.chunk_size = chunk_size
        self.usage = usage or Usage(11, 7)
        self.fail_after = fail_after
        self.error = error or RuntimeError("stream broke")
        self.chat_calls: list[list[ChatMessage]] = []
        self.stream_calls: list[list[ChatMessage]] = []

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        self.chat_calls.append(list(messages))
        return ChatResult(self.text, "fake", self.usage)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]:
        self.stream_calls.append(list(messages))
        size = self.chunk_size
        pieces = [self.text[i : i + size] for i in range(0, len(self.text), size)]
        for n, piece in enumerate(pieces):
            if self.fail_after is not None and n >= self.fail_after:
                raise self.error
            yield ChatDelta(text=piece)
        if self.fail_after is not None and self.fail_after >= len(pieces):
            raise self.error
        yield ChatDelta(usage=self.usage, model="fake")


@dataclass(slots=True)
class FakeChunk:
    id: ChunkId
    text: str
    embedded_text: str
    vector: list[float]
    title: str = "Doc"
    heading_path: tuple[str, ...] = ()
    document_id: DocumentId = field(default_factory=lambda: DocumentId(UUID(int=1)))


@dataclass(slots=True)
class FakeCollection:
    space: EmbeddingSpace
    settings: CollectionSettings = field(default_factory=CollectionSettings)
    chunks: list[FakeChunk] = field(default_factory=list)
    chunk_count: int | None = None

    def add(self, text: str, *, title: str = "Doc", heading_path: tuple[str, ...] = ()) -> ChunkId:
        n = len(self.chunks) + 1
        chunk = FakeChunk(
            ChunkId(UUID(int=n)),
            text,
            text,
            l2_normalise(hash_vector(text)),
            title,
            heading_path,
            DocumentId(UUID(int=1000 + n)),
        )
        self.chunks.append(chunk)
        return chunk.id


class _Snapshot:
    def __init__(self, store: FakeSearchStore) -> None:
        self._store = store

    async def collection(self) -> CollectionView:
        self._store.events.append("collection")
        col = self._store.collection
        count = col.chunk_count if col.chunk_count is not None else len(col.chunks)
        return CollectionView(CollectionId(UUID(int=99)), col.settings, col.space, count)

    async def lexical(self, query: LexicalQuery, filters: SearchFilters) -> list[LegHit]:
        self._store.events.append("lexical")
        self._store.lexical_limits.append(query.limit)
        self._store.lexical_queries.append(query)
        q = words(query.text)
        scored = [(len(q & words(c.text)), c) for c in self._store.collection.chunks]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: (-sc[0], str(sc[1].id)))
        return [LegHit(c.id, float(s)) for s, c in scored[: query.limit]]

    async def vector(self, query: VectorQuery, filters: SearchFilters) -> list[LegHit]:
        self._store.events.append("vector")
        self._store.vector_calls.append(query.exact)
        scored = [
            (cosine_similarity(query.vector, c.vector), c) for c in self._store.collection.chunks
        ]
        scored.sort(key=lambda sc: (-sc[0], str(sc[1].id)))
        limit = query.limit
        if not query.exact and self._store.hnsw_rows is not None:
            limit = min(limit, self._store.hnsw_rows)
        return [LegHit(c.id, s) for s, c in scored[:limit]]

    async def hits(self, chunk_ids: Sequence[ChunkId]) -> list[Hit]:
        self._store.events.append("hits")
        by_id = {c.id: c for c in self._store.collection.chunks}
        return [
            Hit(
                chunk_id=c.id,
                document_id=c.document_id,
                title=c.title,
                heading_path=c.heading_path,
                text=c.text,
                embedded_text=c.embedded_text,
                page_from=None,
                page_to=None,
            )
            for c in (by_id[i] for i in chunk_ids)
        ]

    async def similarities(
        self, chunk_ids: Sequence[ChunkId], vector: Sequence[float]
    ) -> dict[ChunkId, float]:
        self._store.events.append("similarities")
        by_id = {c.id: c for c in self._store.collection.chunks}
        return {i: cosine_similarity(vector, by_id[i].vector) for i in chunk_ids}


class FakeSearchStore:
    def __init__(self, collection: FakeCollection, events: list[str] | None = None) -> None:
        self.collection = collection
        self.events = events if events is not None else []
        self.vector_calls: list[bool] = []
        self.lexical_limits: list[int] = []
        self.lexical_queries: list[LexicalQuery] = []
        self.hnsw_rows: int | None = None

    @asynccontextmanager
    async def snapshot(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> AsyncIterator[SearchSnapshot]:
        self.events.append("enter")
        try:
            yield _Snapshot(self)
        finally:
            self.events.append("exit")


def make_space(calibrated_threshold: float | None = None) -> EmbeddingSpace:
    return EmbeddingSpace(
        1, "fake", "fake", "fake", DIMS, calibrated_threshold=calibrated_threshold
    )


EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(slots=True)
class _Col:
    tenant: TenantId
    id: CollectionId
    name: str
    settings: CollectionSettings
    active_space_id: int
    target_space_id: int | None = None
    chunk_count: int = 0
    reembed: dict[str, Any] | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    status: str = "active"


@dataclass(slots=True)
class _Doc:
    id: DocumentId
    collection_id: CollectionId
    external_id: str
    title: str
    metadata: dict[str, Any]
    status: DocumentStatus = DocumentStatus.QUEUED
    error: dict[str, Any] | None = None
    current_version_id: VersionId | None = None


@dataclass(slots=True)
class _Ver:
    id: VersionId
    document_id: DocumentId
    version: int
    source_sha256: bytes
    media_type: str
    size: int
    pages: int | None
    indexed_title: str
    chunking_hash: str
    status: VersionStatus = VersionStatus.PROCESSING
    error: dict[str, Any] | None = None
    elements: list[Element] | None = None


@dataclass(slots=True)
class _Chunk:
    id: ChunkId
    collection_id: CollectionId
    document_id: DocumentId
    content_hash: str
    embedded_text: str
    language: Language


@dataclass(slots=True)
class _Job:
    tenant: TenantId
    id: JobId
    kind: str
    args: dict[str, Any]
    delay: timedelta
    status: str = "queued"
    attempts: int = 0
    last_error: str | None = None
    progress: dict[str, Any] | None = None


class FakeDb:
    def __init__(self) -> None:
        self.tenants: dict[TenantId, Tenant] = {}
        self.cols: dict[tuple[TenantId, CollectionId], _Col] = {}
        self.docs: dict[tuple[TenantId, DocumentId], _Doc] = {}
        self.versions: dict[tuple[TenantId, VersionId], _Ver] = {}
        self.blobs: dict[tuple[TenantId, VersionId], bytes] = {}
        self.chunks: dict[tuple[TenantId, ChunkId], _Chunk] = {}
        self.members: dict[tuple[TenantId, VersionId], list[Membership]] = {}
        self.vectors: dict[tuple[TenantId, ChunkId, int], list[float]] = {}
        self.jobs: dict[tuple[TenantId, JobId], _Job] = {}
        self.spaces: dict[int, EmbeddingSpace] = {}
        self.hooks: dict[str, Callable[[], None]] = {}
        self._next = 10_000

    def new_id(self) -> UUID:
        self._next += 1
        return UUID(int=self._next)

    def hook(self, name: str) -> None:
        fn = self.hooks.get(name)
        if fn is not None:
            fn()

    def add_space(self, space: EmbeddingSpace) -> EmbeddingSpace:
        self.spaces[space.id] = space
        return space

    def add_tenant(self, quotas: Quotas | None = None) -> TenantId:
        tenant = TenantId(self.new_id())
        self.tenants[tenant] = Tenant(tenant, "t", quotas or Quotas())
        return tenant

    def col(self, tenant: TenantId, collection_id: CollectionId) -> _Col:
        return self.cols[(tenant, collection_id)]

    def jobs_of(self, tenant: TenantId, kind: str | None = None) -> list[_Job]:
        return [
            j for (t, _), j in self.jobs.items() if t == tenant and (kind is None or j.kind == kind)
        ]

    def current_chunks(self, tenant: TenantId, collection_id: CollectionId) -> set[ChunkId]:
        out: set[ChunkId] = set()
        for (t, _), doc in self.docs.items():
            if t == tenant and doc.collection_id == collection_id and doc.current_version_id:
                out |= {m.chunk_id for m in self.members.get((t, doc.current_version_id), [])}
        return out

    def search_view(self, tenant: TenantId, collection_id: CollectionId) -> dict[ChunkId, int]:
        space = self.col(tenant, collection_id).active_space_id
        view = {}
        for chunk_id in self.current_chunks(tenant, collection_id):
            assert (tenant, chunk_id, space) in self.vectors, "current chunk lacks active vector"
            view[chunk_id] = space
        return view

    def needed_spaces(self, tenant: TenantId, collection_id: CollectionId) -> list[int]:
        col = self.col(tenant, collection_id)
        return [s for s in (col.active_space_id, col.target_space_id) if s is not None]

    def insert_job(self, tenant: TenantId, job: NewJob) -> JobId:
        job_id = JobId(self.new_id())
        self.jobs[(tenant, job_id)] = _Job(tenant, job_id, job.kind, dict(job.args), job.delay)
        return job_id


def _col_record(c: _Col) -> CollectionRecord:
    return CollectionRecord(
        c.id,
        c.tenant,
        c.name,
        c.settings,
        c.active_space_id,
        c.target_space_id,
        c.chunk_count,
        c.reembed,
        c.webhook_url,
        c.webhook_secret,
        c.status,
        EPOCH,
        EPOCH,
    )


class FakeCollections:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    async def create(self, tenant: TenantId, new: NewCollection) -> CollectionRecord:
        return self.create_now(tenant, new)

    def create_now(self, tenant: TenantId, new: NewCollection) -> CollectionRecord:
        if any(t == tenant and c.name == new.name for (t, _), c in self.db.cols.items()):
            raise ConflictError("already exists")
        cid = CollectionId(self.db.new_id())
        self.db.cols[(tenant, cid)] = _Col(
            tenant,
            cid,
            new.name,
            new.settings,
            new.space_id,
            webhook_url=new.webhook_url,
            webhook_secret=new.webhook_secret,
        )
        return _col_record(self.db.cols[(tenant, cid)])

    def _get(self, tenant: TenantId, cid: CollectionId, *, include_deleting: bool = False) -> _Col:
        c = self.db.cols.get((tenant, cid))
        if c is None or (c.status != "active" and not include_deleting):
            raise NotFoundError("collection not found")
        return c

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, *, include_deleting: bool = False
    ) -> CollectionRecord:
        return _col_record(self._get(tenant, collection_id, include_deleting=include_deleting))

    async def list_page(
        self, tenant: TenantId, after: CollectionId | None, limit: int
    ) -> list[CollectionRecord]:
        cols = sorted(
            (c for (t, _), c in self.db.cols.items() if t == tenant and c.status == "active"),
            key=lambda c: c.id,
        )
        return [_col_record(c) for c in cols if after is None or c.id > after][:limit]

    async def count(self, tenant: TenantId) -> int:
        return sum(1 for (t, _) in self.db.cols if t == tenant)

    async def total_chunks(self, tenant: TenantId) -> int:
        return sum(c.chunk_count for (t, _), c in self.db.cols.items() if t == tenant)

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        settings: CollectionSettings,
        *,
        webhook: tuple[str | None, str | None] | None = None,
        jobs: Sequence[NewJob] = (),
    ) -> tuple[CollectionRecord, list[JobId]]:
        c = self._get(tenant, collection_id)
        c.settings = replace(
            settings, language=c.settings.language, embedding_space=c.settings.embedding_space
        )
        if webhook is not None:
            c.webhook_url, c.webhook_secret = webhook
        ids = [self.db.insert_job(tenant, j) for j in jobs]
        return _col_record(c), ids

    async def set_target_space(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int | None
    ) -> CollectionRecord:
        c = self._get(tenant, collection_id)
        c.target_space_id = space_id
        c.reembed = (
            None
            if space_id is None
            else {"target_space_id": space_id, "done": 0, "total": 0, "state": "running"}
        )
        return _col_record(c)

    async def set_reembed_progress(
        self, tenant: TenantId, collection_id: CollectionId, done: int, total: int
    ) -> None:
        c = self.db.cols.get((tenant, collection_id))
        if c is not None and c.target_space_id is not None:
            c.reembed = {
                "target_space_id": c.target_space_id,
                "done": done,
                "total": total,
                "state": "running",
            }

    async def mark_deleting(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> list[JobId]:
        self._get(tenant, collection_id).status = "deleting"
        return [self.db.insert_job(tenant, j) for j in jobs]

    async def delete_rows(self, tenant: TenantId, collection_id: CollectionId) -> None:
        self.db.cols.pop((tenant, collection_id), None)
        for key, doc in list(self.db.docs.items()):
            if key[0] == tenant and doc.collection_id == collection_id:
                _delete_document(self.db, tenant, doc.id)

    async def set_language(
        self, tenant: TenantId, collection_id: CollectionId, language: Language
    ) -> None:
        c = self._get(tenant, collection_id)
        c.settings = replace(c.settings, language=language)
        for (t, _), chunk in self.db.chunks.items():
            if t == tenant and chunk.collection_id == collection_id:
                chunk.language = language


def _delete_document(db: FakeDb, tenant: TenantId, document_id: DocumentId) -> None:
    db.docs.pop((tenant, document_id), None)
    for (t, vid), v in list(db.versions.items()):
        if t == tenant and v.document_id == document_id:
            db.versions.pop((t, vid))
            db.blobs.pop((t, vid), None)
            db.members.pop((t, vid), None)
    for (t, cid), c in list(db.chunks.items()):
        if t == tenant and c.document_id == document_id:
            db.chunks.pop((t, cid))
            for key in [k for k in db.vectors if k[0] == t and k[1] == cid]:
                db.vectors.pop(key)


class FakeDocuments:
    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.usage: dict[TenantId, int] = {}

    def _doc(self, tenant: TenantId, document_id: DocumentId) -> _Doc:
        doc = self.db.docs.get((tenant, document_id))
        if doc is None:
            raise NotFoundError("document not found")
        return doc

    def _ver(self, tenant: TenantId, version_id: VersionId) -> _Ver:
        v = self.db.versions.get((tenant, version_id))
        if v is None:
            raise NotFoundError("document version not found")
        return v

    def _versions_of(self, tenant: TenantId, document_id: DocumentId) -> list[_Ver]:
        return sorted(
            (
                v
                for (t, _), v in self.db.versions.items()
                if t == tenant and v.document_id == document_id
            ),
            key=lambda v: v.version,
        )

    def _record(self, tenant: TenantId, doc: _Doc) -> DocumentRecord:
        cur = (
            self.db.versions.get((tenant, doc.current_version_id))
            if doc.current_version_id
            else None
        )
        return DocumentRecord(
            doc.id,
            doc.collection_id,
            doc.external_id,
            doc.title,
            doc.metadata,
            doc.status,
            doc.error,
            doc.current_version_id,
            cur.version if cur else None,
            EPOCH,
            EPOCH,
        )

    def _vrecord(self, tenant: TenantId, v: _Ver) -> VersionRecord:
        doc = self._doc(tenant, v.document_id)
        return VersionRecord(
            v.id,
            v.document_id,
            doc.collection_id,
            v.version,
            v.source_sha256,
            v.media_type,
            v.size,
            v.pages,
            v.indexed_title,
            v.chunking_hash,
            v.status,
            v.error,
            (tenant, v.id) in self.db.blobs,
            EPOCH,
            EPOCH,
        )

    def _new_version(self, tenant: TenantId, doc: _Doc, **fields: Any) -> _Ver:
        number = max((v.version for v in self._versions_of(tenant, doc.id)), default=0) + 1
        v = _Ver(VersionId(self.db.new_id()), doc.id, number, **fields)
        self.db.versions[(tenant, v.id)] = v
        return v

    async def upload(self, tenant: TenantId, up: NewUpload) -> UploadResult:
        col = self.db.cols.get((tenant, up.collection_id))
        if col is None or col.status != "active":
            raise NotFoundError("collection not found")
        doc = next(
            (
                d
                for (t, _), d in self.db.docs.items()
                if t == tenant
                and d.collection_id == up.collection_id
                and up.external_id is not None
                and d.external_id == up.external_id
            ),
            None,
        )
        if doc is None:
            did = DocumentId(self.db.new_id())
            doc = _Doc(did, up.collection_id, up.external_id or str(did), up.title, up.metadata)
            self.db.docs[(tenant, did)] = doc
        cur = (
            self.db.versions.get((tenant, doc.current_version_id))
            if doc.current_version_id
            else None
        )
        if cur is not None and (cur.source_sha256, cur.indexed_title, cur.chunking_hash) == (
            up.source_sha256,
            up.title,
            up.chunking_hash,
        ):
            doc.metadata = up.metadata
            return UploadResult(doc.id, unchanged=True)
        v = self._new_version(
            tenant,
            doc,
            source_sha256=up.source_sha256,
            media_type=up.media_type,
            size=len(up.data),
            pages=up.pages,
            indexed_title=up.title,
            chunking_hash=up.chunking_hash,
        )
        self.db.blobs[(tenant, v.id)] = up.data
        doc.title, doc.metadata, doc.status, doc.error = (
            up.title,
            up.metadata,
            DocumentStatus.QUEUED,
            None,
        )
        job = self.db.insert_job(
            tenant,
            NewJob("ingest", _ingest_args(up.collection_id, doc.id, v.id), up.ingest_delay),
        )
        self.usage[tenant] = self.usage.get(tenant, 0) + len(up.data)
        return UploadResult(doc.id, False, v.id, v.version, job, self.usage[tenant])

    async def reversion(
        self, tenant: TenantId, document_id: DocumentId, chunking_hash: str
    ) -> UploadResult | None:
        doc = self._doc(tenant, document_id)
        if doc.current_version_id is None:
            return None
        latest = self._versions_of(tenant, document_id)[-1]
        if latest.status is VersionStatus.PROCESSING or (
            latest.status is VersionStatus.READY and latest.chunking_hash == chunking_hash
        ):
            return None
        cur = self._ver(tenant, doc.current_version_id)
        if cur.elements is None:
            return None
        v = self._new_version(
            tenant,
            doc,
            source_sha256=cur.source_sha256,
            media_type=cur.media_type,
            size=cur.size,
            pages=cur.pages,
            indexed_title=cur.indexed_title,
            chunking_hash=chunking_hash,
            elements=list(cur.elements),
        )
        if (tenant, cur.id) in self.db.blobs:
            self.db.blobs[(tenant, v.id)] = self.db.blobs[(tenant, cur.id)]
        doc.status, doc.error = DocumentStatus.QUEUED, None
        job = self.db.insert_job(
            tenant, NewJob("ingest", _ingest_args(doc.collection_id, doc.id, v.id))
        )
        return UploadResult(doc.id, False, v.id, v.version, job)

    async def uploaded_bytes_today(self, tenant: TenantId) -> int:
        return self.usage.get(tenant, 0)

    async def count(self, tenant: TenantId) -> int:
        return sum(1 for (t, _) in self.db.docs if t == tenant)

    async def find(
        self, tenant: TenantId, collection_id: CollectionId, external_id: str
    ) -> DocumentRecord | None:
        for (t, _), d in self.db.docs.items():
            if t == tenant and d.collection_id == collection_id and d.external_id == external_id:
                return self._record(tenant, d)
        return None

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> DocumentRecord:
        doc = self._doc(tenant, document_id)
        if doc.collection_id != collection_id:
            raise NotFoundError("document not found")
        return self._record(tenant, doc)

    async def list_page(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        after: DocumentId | None,
        limit: int,
    ) -> list[DocumentRecord]:
        docs = sorted(
            (
                d
                for (t, _), d in self.db.docs.items()
                if t == tenant and d.collection_id == collection_id
            ),
            key=lambda d: d.id,
        )
        return [self._record(tenant, d) for d in docs if after is None or d.id > after][:limit]

    async def versions(self, tenant: TenantId, document_id: DocumentId) -> list[VersionRecord]:
        return [self._vrecord(tenant, v) for v in self._versions_of(tenant, document_id)]

    async def version(self, tenant: TenantId, version_id: VersionId) -> VersionRecord:
        return self._vrecord(tenant, self._ver(tenant, version_id))

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> DocumentRecord:
        doc = self._doc(tenant, document_id)
        doc.metadata = metadata if metadata is not None else doc.metadata
        doc.title = title if title is not None else doc.title
        return self._record(tenant, doc)

    def _is_latest(self, tenant: TenantId, version_id: VersionId) -> bool:
        v = self._ver(tenant, version_id)
        return all(o.version <= v.version for o in self._versions_of(tenant, v.document_id))

    async def set_status(
        self,
        tenant: TenantId,
        document_id: DocumentId,
        status: DocumentStatus,
        error: dict[str, Any] | None = None,
        *,
        version_id: VersionId | None = None,
    ) -> None:
        doc = self._doc(tenant, document_id)
        if version_id is not None and not self._is_latest(tenant, version_id):
            return
        doc.status, doc.error = status, error

    async def store_elements(
        self,
        tenant: TenantId,
        version_id: VersionId,
        elements: list[Element],
        pages: int | None = None,
    ) -> None:
        v = self._ver(tenant, version_id)
        v.elements = list(elements)
        v.pages = pages if pages is not None else v.pages

    async def elements(self, tenant: TenantId, version_id: VersionId) -> list[Element]:
        v = self._ver(tenant, version_id)
        if v.elements is None:
            raise NotFoundError("document version has no parsed elements")
        return list(v.elements)

    async def fail_version(
        self, tenant: TenantId, version_id: VersionId, error: dict[str, Any]
    ) -> None:
        v = self._ver(tenant, version_id)
        v.status, v.error = VersionStatus.FAILED, error
        if self._is_latest(tenant, version_id):
            doc = self._doc(tenant, v.document_id)
            doc.status, doc.error = DocumentStatus.FAILED, error

    async def original(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        version: int | None = None,
    ) -> tuple[str, bytes]:
        doc = self._doc(tenant, document_id)
        match = [
            v
            for v in self._versions_of(tenant, document_id)
            if (version is None and v.id == doc.current_version_id) or v.version == version
        ]
        if not match:
            raise NotFoundError("document version not found")
        data = self.db.blobs.get((tenant, match[0].id))
        if data is None:
            raise NotFoundError("original not retained", code="original_not_retained")
        return match[0].media_type, data

    async def source(self, tenant: TenantId, version_id: VersionId) -> bytes:
        data = self.db.blobs.get((tenant, version_id))
        if data is None:
            raise NotFoundError("original not retained", code="original_not_retained")
        return data

    async def delete_original(self, tenant: TenantId, version_id: VersionId) -> None:
        self.db.blobs.pop((tenant, version_id), None)

    async def delete(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> None:
        doc = self._doc(tenant, document_id)
        if doc.current_version_id is not None:
            col = self.db.col(tenant, collection_id)
            n = len({m.chunk_id for m in self.db.members.get((tenant, doc.current_version_id), [])})
            col.chunk_count = max(col.chunk_count - n, 0)
        _delete_document(self.db, tenant, document_id)


def _ingest_args(cid: CollectionId, did: DocumentId, vid: VersionId) -> dict[str, Any]:
    return {"collection_id": str(cid), "document_id": str(did), "version_id": str(vid)}


class FakeIndex:
    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.swaps: list[VersionId] = []
        self.switches: list[CollectionId] = []

    async def existing_chunks(
        self, tenant: TenantId, document_id: DocumentId, hashes: Sequence[str]
    ) -> dict[str, ChunkId]:
        wanted = set(hashes)
        return {
            c.content_hash: c.id
            for (t, _), c in self.db.chunks.items()
            if t == tenant and c.document_id == document_id and c.content_hash in wanted
        }

    async def insert_chunks(self, tenant: TenantId, batch: ChunkBatch) -> dict[str, ChunkId]:
        existing = await self.existing_chunks(
            tenant, batch.document_id, [d.content_hash for d in batch.drafts]
        )
        for d in batch.drafts:
            if d.content_hash not in existing:
                cid = ChunkId(self.db.new_id())
                self.db.chunks[(tenant, cid)] = _Chunk(
                    cid,
                    batch.collection_id,
                    batch.document_id,
                    d.content_hash,
                    d.embedded_text,
                    batch.language,
                )
                existing[d.content_hash] = cid
        return {d.content_hash: existing[d.content_hash] for d in batch.drafts}

    async def insert_vectors(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        space: EmbeddingSpace,
        vectors: Sequence[tuple[ChunkId, Sequence[float]]],
    ) -> None:
        for chunk_id, vector in vectors:
            if len(vector) != space.dimensions:
                raise ValidationError("embedding", "dimension", code="embedding_dimension_mismatch")
            if (tenant, chunk_id) not in self.db.chunks:
                raise NotFoundError("chunk not found")
            self.db.vectors.setdefault((tenant, chunk_id, space.id), l2_normalise(vector))

    def _missing(
        self, tenant: TenantId, collection_id: CollectionId, chunk_ids: Sequence[ChunkId]
    ) -> dict[int, list[ChunkId]]:
        out: dict[int, list[ChunkId]] = {}
        for space in self.db.needed_spaces(tenant, collection_id):
            lacking = sorted({c for c in chunk_ids if (tenant, c, space) not in self.db.vectors})
            if lacking:
                out[space] = lacking
        return out

    async def missing_vectors(
        self, tenant: TenantId, collection_id: CollectionId, chunk_ids: Sequence[ChunkId]
    ) -> dict[int, list[ChunkId]]:
        return self._missing(tenant, collection_id, chunk_ids)

    async def current_chunk_count(self, tenant: TenantId, document_id: DocumentId) -> int:
        doc = self.db.docs.get((tenant, document_id))
        if doc is None or doc.current_version_id is None:
            return 0
        return len({m.chunk_id for m in self.db.members.get((tenant, doc.current_version_id), [])})

    async def swap(  # noqa: PLR0913 - mirrors IndexRepository.swap
        self,
        tenant: TenantId,
        document_id: DocumentId,
        version_id: VersionId,
        memberships: Sequence[Membership],
        *,
        jobs: Sequence[NewJob] = (),
        on_current: Sequence[NewJob] = (),
    ) -> SwapResult:
        self.db.hook("before_swap")
        doc = self.db.docs.get((tenant, document_id))
        new = self.db.versions.get((tenant, version_id))
        if doc is None or new is None or new.document_id != document_id:
            raise NotFoundError("document version not found")
        col = self.db.col(tenant, doc.collection_id)
        chunk_ids = list(dict.fromkeys(m.chunk_id for m in memberships))
        if self._missing(tenant, col.id, chunk_ids):
            raise ConflictError("vectors missing", code="vectors_missing")
        self.db.members[(tenant, version_id)] = [
            m for m in memberships if m.version_id == version_id
        ]
        old = (
            self.db.versions.get((tenant, doc.current_version_id))
            if doc.current_version_id
            else None
        )
        current = old is None or new.version > old.version
        if current:
            new.status = VersionStatus.READY
            old_n = 0
            if old is not None:
                old.status = VersionStatus.SUPERSEDED
                old_n = len({m.chunk_id for m in self.db.members.get((tenant, old.id), [])})
            doc.current_version_id, doc.status, doc.error = version_id, DocumentStatus.READY, None
            col.chunk_count = max(col.chunk_count + len(set(chunk_ids)) - old_n, 0)
        else:
            new.status = VersionStatus.SUPERSEDED
        ids = [self.db.insert_job(tenant, j) for j in (*jobs, *(on_current if current else ()))]
        self.swaps.append(version_id)
        return SwapResult(current, col.chunk_count, tuple(ids))

    async def gc_document(
        self, tenant: TenantId, document_id: DocumentId, *, delete_superseded_originals: bool
    ) -> GcStats:
        if (tenant, document_id) not in self.db.docs:
            raise NotFoundError("document not found")
        versions = [
            v
            for (t, _), v in self.db.versions.items()
            if t == tenant and v.document_id == document_id
        ]
        memberships = 0
        for v in versions:
            if v.status in (VersionStatus.SUPERSEDED, VersionStatus.FAILED):
                memberships += len(self.db.members.pop((tenant, v.id), []))
        chunks = 0
        if not any(v.status is VersionStatus.PROCESSING for v in versions):
            referenced = {
                m.chunk_id for (t, _), ms in self.db.members.items() if t == tenant for m in ms
            }
            for (t, cid), c in list(self.db.chunks.items()):
                if t == tenant and c.document_id == document_id and cid not in referenced:
                    self.db.chunks.pop((t, cid))
                    for key in [k for k in self.db.vectors if k[0] == t and k[1] == cid]:
                        self.db.vectors.pop(key)
                    chunks += 1
        blobs = 0
        if delete_superseded_originals:
            for v in versions:
                if v.status is VersionStatus.SUPERSEDED and self.db.blobs.pop((tenant, v.id), None):
                    blobs += 1
        return GcStats(memberships, chunks, blobs)

    async def gc_space(self, tenant: TenantId, collection_id: CollectionId, space_id: int) -> int:
        col = self.db.cols.get((tenant, collection_id))
        if col is None or space_id in (col.active_space_id, col.target_space_id):
            return 0
        doomed = [
            k
            for k in self.db.vectors
            if k[0] == tenant
            and k[2] == space_id
            and self.db.chunks[(tenant, k[1])].collection_id == collection_id
        ]
        for k in doomed:
            self.db.vectors.pop(k)
        return len(doomed)

    def _current_in(self, tenant: TenantId, collection_id: CollectionId) -> list[ChunkId]:
        return sorted(self.db.current_chunks(tenant, collection_id))

    async def reembed_batch(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int, limit: int
    ) -> list[tuple[ChunkId, str]]:
        lacking = [
            c
            for c in self._current_in(tenant, collection_id)
            if (tenant, c, space_id) not in self.db.vectors
        ]
        return [(c, self.db.chunks[(tenant, c)].embedded_text) for c in lacking[:limit]]

    async def reembed_counts(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> tuple[int, int]:
        current = self._current_in(tenant, collection_id)
        done = sum(1 for c in current if (tenant, c, space_id) in self.db.vectors)
        return done, len(current)

    async def switch_space(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> bool:
        self.db.hook("before_switch")
        col = self.db.col(tenant, collection_id)
        target = col.target_space_id
        if target is None:
            return False
        if any(
            (tenant, c, target) not in self.db.vectors
            for c in self._current_in(tenant, collection_id)
        ):
            return False
        col.active_space_id, col.target_space_id = target, None
        col.reembed = {**(col.reembed or {}), "state": "done"}
        for job in jobs:
            self.db.insert_job(tenant, job)
        self.switches.append(collection_id)
        return True


class FakeJobs:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    async def enqueue(self, tenant: TenantId, job: NewJob) -> JobId:
        return self.db.insert_job(tenant, job)

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        j = self.db.jobs.get((tenant, job_id))
        if j is None:
            raise NotFoundError("job not found")
        return JobRecord(
            j.id,
            j.tenant,
            j.kind,
            j.args,
            j.status,
            j.attempts,
            j.last_error,
            j.progress,
            EPOCH,
            EPOCH,
            None,
            None,
            EPOCH,
        )

    async def active_for_collection(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> list[JobRecord]:
        return [
            await self.get(tenant, j.id)
            for j in self.db.jobs_of(tenant)
            if j.status in ("queued", "running")
            and j.args.get("collection_id") == str(collection_id)
        ]

    def claim(self, tenant: TenantId, kind: str | None = None) -> FakeJobRun | None:
        for j in self.db.jobs_of(tenant, kind):
            if j.status == "queued":
                j.status, j.attempts = "running", j.attempts + 1
                return FakeJobRun(ClaimedJob(j.id, tenant, j.kind), j)
        return None


@dataclass(slots=True)
class FakeJobRun:
    job: ClaimedJob
    row: _Job
    progress_calls: list[tuple[int, int]] = field(default_factory=list)

    async def progress(self, done: int, total: int) -> None:
        self.progress_calls.append((done, total))
        self.row.progress = {"done": done, "total": total}


class FakeSpaces:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    async def list_spaces(self) -> list[EmbeddingSpace]:
        return [self.db.spaces[k] for k in sorted(self.db.spaces)]

    async def list_spaces_page(self, after: int | None, limit: int) -> list[EmbeddingSpace]:
        ordered = [self.db.spaces[k] for k in sorted(self.db.spaces)]
        return [s for s in ordered if after is None or s.id > after][:limit]

    async def get(self, name: str) -> EmbeddingSpace:
        for s in self.db.spaces.values():
            if s.name == name:
                return s
        raise NotFoundError(f"embedding space {name!r} not found")

    async def get_by_id(self, space_id: int) -> EmbeddingSpace:
        if space_id not in self.db.spaces:
            raise NotFoundError("embedding space not found")
        return self.db.spaces[space_id]


class FakeTenants:
    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.deleted: list[TenantId] = []

    async def get_tenant(self, tenant: TenantId) -> Tenant:
        if tenant not in self.db.tenants:
            raise NotFoundError("tenant not found")
        return self.db.tenants[tenant]

    async def delete_tenant_data(self, tenant: TenantId) -> None:
        self.deleted.append(tenant)
        self.db.tenants.pop(tenant, None)
        for store in (
            self.db.cols,
            self.db.docs,
            self.db.versions,
            self.db.blobs,
            self.db.chunks,
            self.db.members,
            self.db.jobs,
        ):
            for key in [k for k in store if k[0] == tenant]:
                store.pop(key)  # type: ignore[attr-defined]
        for key in [k for k in self.db.vectors if k[0] == tenant]:
            self.db.vectors.pop(key)


IMAGE_BLOCK = "[image]"


class FakeParser:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def parse(self, media_type: str, data: bytes) -> ParsedDocument:
        self.calls += 1
        if self.error is not None:
            raise self.error
        blocks = [b.strip() for b in data.decode().split("\n\n") if b.strip()]
        elements: list[Element] = []
        images: list[PageImage] = []
        for n, block in enumerate(blocks):
            if block == IMAGE_BLOCK:
                elements.append(Element(ElementKind.IMAGE, "", n, page=n + 1, page_to=n + 1))
                images.append(PageImage(n, n + 1, "image/png", f"image {n}".encode()))
            else:
                elements.append(Element(ElementKind.PARAGRAPH, block, n, page=n + 1, page_to=n + 1))
        return ParsedDocument(elements, images)


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class FakeTokenizers:
    def for_tokenizer(self, tokenizer_id: str) -> TokenCounter:
        return WordCounter()
