# SPDX-License-Identifier: Apache-2.0
"""Ports that the use cases need from the outside world."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from footnote.domain.answer import Usage
from footnote.domain.deliveries import AttemptRecord, DeliveryRecord
from footnote.domain.documents import (
    ChunkBatch,
    ChunkDetail,
    ChunkId,
    CollectionId,
    CollectionRecord,
    DocumentId,
    DocumentRecord,
    DocumentStatus,
    Element,
    GcStats,
    InspectedFile,
    JobId,
    Membership,
    NewCollection,
    NewUpload,
    SwapResult,
    UploadResult,
    VersionId,
    VersionRecord,
)
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec
from footnote.domain.jobs import ClaimedJob, JobRecord, NewJob
from footnote.domain.retrieval import Hit, LegHit
from footnote.domain.settings import CollectionSettings, Language, LexicalMatch, LexicalRank
from footnote.domain.tenancy import (
    ApiKeyRecord,
    ExportRecord,
    IssuedKey,
    Quotas,
    Tenant,
    TenantId,
    TenantUsage,
)


class TokenCounter(Protocol):
    """Counts the tokens of a text."""

    def count(self, text: str) -> int: ...


class Embedder(Protocol):
    """Turns texts into vectors of one embedding space."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class EmbedderFactory(Protocol):
    """Returns the embedder of a registered space."""

    def for_space(self, space: EmbeddingSpace) -> Embedder: ...


class Reranker(Protocol):
    """Scores query and passage pairs."""

    @property
    def model_key(self) -> str: ...

    @property
    def calibrated_threshold(self) -> float | None: ...

    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class ChatDelta:
    text: str = ""
    usage: Usage | None = None
    model: str | None = None


class LLM(Protocol):
    """Chat model with an OpenAI-compatible API."""

    @property
    def configured(self) -> bool: ...

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResult: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]: ...


class Parser(Protocol):
    """Turns an original file into elements."""

    async def parse(self, media_type: str, data: bytes) -> list[Element]: ...


@dataclass(frozen=True, slots=True)
class CollectionView:
    id: CollectionId
    settings: CollectionSettings
    active_space: EmbeddingSpace
    chunk_count: int


@dataclass(frozen=True, slots=True)
class SearchFilters:
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_ids: tuple[DocumentId, ...] = ()
    created_after: str | None = None
    created_before: str | None = None


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    text: str
    match: LexicalMatch
    rank: LexicalRank
    limit: int


@dataclass(frozen=True, slots=True)
class VectorQuery:
    vector: Sequence[float]
    limit: int
    exact: bool


class SearchSnapshot(Protocol):
    """Read-only view of the current versions for one search."""

    async def collection(self) -> CollectionView: ...

    async def lexical(self, query: LexicalQuery, filters: SearchFilters) -> list[LegHit]: ...

    async def vector(self, query: VectorQuery, filters: SearchFilters) -> list[LegHit]: ...

    async def hits(self, chunk_ids: Sequence[ChunkId]) -> list[Hit]: ...

    async def similarities(
        self, chunk_ids: Sequence[ChunkId], vector: Sequence[float]
    ) -> dict[ChunkId, float]: ...


class SearchStore(Protocol):
    """Opens search snapshots."""

    def snapshot(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> AbstractAsyncContextManager[SearchSnapshot]: ...


class Clock(Protocol):
    """Time source."""

    def now(self) -> float: ...

    def monotonic(self) -> float: ...


class TokenCounterFactory(Protocol):
    """Returns the token counter of a tokenizer id."""

    def for_tokenizer(self, tokenizer_id: str) -> TokenCounter: ...


class CollectionRepository(Protocol):
    """Stores the collections of a tenant."""

    async def create(self, tenant: TenantId, new: NewCollection) -> CollectionRecord: ...

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, *, include_deleting: bool = False
    ) -> CollectionRecord: ...

    async def list_page(
        self, tenant: TenantId, after: CollectionId | None, limit: int
    ) -> list[CollectionRecord]: ...

    async def count(self, tenant: TenantId) -> int: ...

    async def total_chunks(self, tenant: TenantId) -> int: ...

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        settings: CollectionSettings,
        *,
        webhook: tuple[str | None, str | None] | None = None,
        jobs: Sequence[NewJob] = (),
    ) -> tuple[CollectionRecord, list[JobId]]: ...

    async def set_target_space(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int | None
    ) -> CollectionRecord: ...

    async def set_reembed_progress(
        self, tenant: TenantId, collection_id: CollectionId, done: int, total: int
    ) -> None: ...

    async def mark_deleting(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> list[JobId]: ...

    async def delete_rows(self, tenant: TenantId, collection_id: CollectionId) -> None: ...

    async def set_language(
        self, tenant: TenantId, collection_id: CollectionId, language: Language
    ) -> None: ...


class BlobStore(Protocol):
    """Stores original files."""

    async def source(self, tenant: TenantId, version_id: VersionId) -> bytes: ...

    async def delete_original(self, tenant: TenantId, version_id: VersionId) -> None: ...


class DocumentRepository(BlobStore, Protocol):
    """Stores documents, versions, elements and originals."""

    async def upload(self, tenant: TenantId, up: NewUpload) -> UploadResult: ...

    async def reversion(
        self, tenant: TenantId, document_id: DocumentId, chunking_hash: str
    ) -> UploadResult | None: ...

    async def uploaded_bytes_today(self, tenant: TenantId) -> int: ...

    async def count(self, tenant: TenantId) -> int: ...

    async def find(
        self, tenant: TenantId, collection_id: CollectionId, external_id: str
    ) -> DocumentRecord | None: ...

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> DocumentRecord: ...

    async def list_page(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        after: DocumentId | None,
        limit: int,
    ) -> list[DocumentRecord]: ...

    async def versions(self, tenant: TenantId, document_id: DocumentId) -> list[VersionRecord]: ...

    async def version(self, tenant: TenantId, version_id: VersionId) -> VersionRecord: ...

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> DocumentRecord: ...

    async def set_status(
        self,
        tenant: TenantId,
        document_id: DocumentId,
        status: DocumentStatus,
        error: dict[str, Any] | None = None,
        *,
        version_id: VersionId | None = None,
    ) -> None: ...

    async def store_elements(
        self,
        tenant: TenantId,
        version_id: VersionId,
        elements: list[Element],
        pages: int | None = None,
    ) -> None: ...

    async def elements(self, tenant: TenantId, version_id: VersionId) -> list[Element]: ...

    async def fail_version(
        self, tenant: TenantId, version_id: VersionId, error: dict[str, Any]
    ) -> None: ...

    async def original(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        version: int | None = None,
    ) -> tuple[str, bytes]: ...

    async def delete(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> None: ...


class IndexRepository(Protocol):
    """Stores chunks, memberships and vectors, and swaps versions."""

    async def existing_chunks(
        self, tenant: TenantId, document_id: DocumentId, hashes: Sequence[str]
    ) -> dict[str, ChunkId]: ...

    async def insert_chunks(self, tenant: TenantId, batch: ChunkBatch) -> dict[str, ChunkId]: ...

    async def insert_vectors(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        space: EmbeddingSpace,
        vectors: Sequence[tuple[ChunkId, Sequence[float]]],
    ) -> None: ...

    async def missing_vectors(
        self, tenant: TenantId, collection_id: CollectionId, chunk_ids: Sequence[ChunkId]
    ) -> dict[int, list[ChunkId]]: ...

    async def current_chunk_count(self, tenant: TenantId, document_id: DocumentId) -> int: ...

    async def swap(  # noqa: PLR0913 - mirrors the swap transaction's inputs
        self,
        tenant: TenantId,
        document_id: DocumentId,
        version_id: VersionId,
        memberships: Sequence[Membership],
        *,
        jobs: Sequence[NewJob] = (),
        on_current: Sequence[NewJob] = (),
    ) -> SwapResult: ...

    async def gc_document(
        self, tenant: TenantId, document_id: DocumentId, *, delete_superseded_originals: bool
    ) -> GcStats: ...

    async def gc_space(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> int: ...

    async def reembed_batch(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int, limit: int
    ) -> list[tuple[ChunkId, str]]: ...

    async def reembed_counts(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> tuple[int, int]: ...

    async def switch_space(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> bool: ...


class JobQueue(Protocol):
    """Enqueues and reads jobs."""

    async def enqueue(self, tenant: TenantId, job: NewJob) -> JobId: ...

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord: ...

    async def active_for_collection(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> list[JobRecord]: ...


class JobRun(Protocol):
    """Context a job handler receives from the worker."""

    @property
    def job(self) -> ClaimedJob: ...

    async def progress(self, done: int, total: int) -> None: ...


class SpaceReader(Protocol):
    """Reads the registered embedding spaces."""

    async def list_spaces(self) -> list[EmbeddingSpace]: ...

    async def list_spaces_page(self, after: int | None, limit: int) -> list[EmbeddingSpace]: ...

    async def get(self, name: str) -> EmbeddingSpace: ...

    async def get_by_id(self, space_id: int) -> EmbeddingSpace: ...


class SpaceRegistry(SpaceReader, Protocol):
    """Registers and deletes embedding spaces."""

    async def register(self, spec: SpaceSpec) -> EmbeddingSpace: ...

    async def delete(self, name: str) -> None: ...


class TenantReader(Protocol):
    """Reads a tenant."""

    async def get_tenant(self, tenant: TenantId) -> Tenant: ...


class TenantAdmin(TenantReader, Protocol):
    """Administers tenants and keys."""

    async def create_tenant(self, name: str, quotas: Quotas) -> Tenant: ...

    async def list_tenants(self, after: TenantId | None, limit: int) -> list[Tenant]: ...

    async def update_tenant(
        self, tenant: TenantId, *, name: str | None = None, quotas: Quotas | None = None
    ) -> Tenant: ...

    async def mark_deleting(self, tenant: TenantId, jobs: Sequence[NewJob] = ()) -> list[JobId]: ...

    async def delete_tenant_data(self, tenant: TenantId) -> None: ...

    async def usage(self, tenant: TenantId) -> TenantUsage: ...

    async def create_key(
        self, tenant: TenantId, name: str, expires_at: datetime | None = None
    ) -> IssuedKey: ...

    async def list_keys(
        self, tenant: TenantId, after: UUID | None, limit: int
    ) -> list[ApiKeyRecord]: ...

    async def revoke_key(self, tenant: TenantId, key_id: UUID) -> None: ...

    async def rotate_key(self, tenant: TenantId, key_id: UUID, grace: timedelta) -> IssuedKey: ...


class KeyResolver(Protocol):
    """Resolves API keys to tenants."""

    async def resolve(self, key: str) -> tuple[TenantId, UUID]: ...


class WebhookDeliveryLog(Protocol):
    """Records webhook deliveries."""

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int: ...

    async def record_attempt(self, tenant: TenantId, attempt: AttemptRecord) -> None: ...

    async def list_for_collection(
        self, tenant: TenantId, collection_id: CollectionId, limit: int = 50
    ) -> list[DeliveryRecord]: ...


class UploadInspector(Protocol):
    """Checks an upload before it is stored."""

    async def inspect(self, data: bytes, filename: str) -> InspectedFile: ...


class HostResolver(Protocol):
    """Resolves a webhook host to addresses."""

    async def resolve(self, host: str, port: int) -> Sequence[str]: ...


class ChunkReader(Protocol):
    """Reads one chunk of a current version."""

    async def chunk(
        self, tenant: TenantId, collection_id: CollectionId, chunk_id: ChunkId
    ) -> ChunkDetail: ...


class TenantExporter(Protocol):
    """Streams the data of a tenant."""

    def export(
        self, tenant: TenantId, *, include_vectors: bool, include_originals: bool
    ) -> AsyncIterator[ExportRecord]: ...
