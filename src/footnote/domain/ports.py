# SPDX-License-Identifier: Apache-2.0
"""Ports: the behaviour the use cases need from the outside world (adapters implement them)."""

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
    """Counts tokens of a text, excluding special tokens."""

    def count(self, text: str) -> int:
        """Number of tokens in `text`."""


class Embedder(Protocol):
    """Turns texts into vectors of one embedding space."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts (embedded text form), one vector per input."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (query prefix applied by the adapter)."""


class EmbedderFactory(Protocol):
    """Returns the embedder serving a registered space."""

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        """Embedder for `space`; raises UnavailableError when it cannot be served."""


class Reranker(Protocol):
    """Cross-encoder scoring of (query, passage) pairs."""

    @property
    def model_key(self) -> str:
        """Identifies the active model file (used to look up its calibrated threshold)."""

    @property
    def calibrated_threshold(self) -> float | None:
        """Manifest-calibrated gate threshold for the active file, if any."""

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw logits, one per passage."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Non-streamed completion; `text` excludes reasoning output."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class ChatDelta:
    """One streamed fragment; the final delta carries usage when the provider sends it."""

    text: str = ""
    usage: Usage | None = None
    model: str | None = None


class LLM(Protocol):
    """OpenAI-compatible chat model."""

    @property
    def configured(self) -> bool:
        """False when no base URL is set (answers return `llm_not_configured`)."""

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """Complete a chat."""

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]:
        """Stream a chat completion."""


class Parser(Protocol):
    """Turns an original file into elements (run in a sandbox by the adapter)."""

    async def parse(self, media_type: str, data: bytes) -> list[Element]:
        """Parse `data`; raises DomainError with codes parse_failed/parse_timeout/…"""


@dataclass(frozen=True, slots=True)
class CollectionView:
    """What a search reads about its collection inside the snapshot."""

    id: CollectionId
    settings: CollectionSettings
    active_space: EmbeddingSpace
    chunk_count: int


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Metadata, document and time filters (validated by the API layer)."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_ids: tuple[DocumentId, ...] = ()
    created_after: str | None = None
    created_before: str | None = None


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    """Parameters of the lexical leg."""

    text: str
    match: LexicalMatch
    rank: LexicalRank
    limit: int


@dataclass(frozen=True, slots=True)
class VectorQuery:
    """Parameters of the vector leg; `exact` selects the btree exact scan over HNSW."""

    vector: Sequence[float]
    limit: int
    exact: bool


class SearchSnapshot(Protocol):
    """One REPEATABLE READ READ ONLY transaction over current versions."""

    async def collection(self) -> CollectionView:
        """The collection as of the snapshot; NotFoundError when absent or deleting."""

    async def lexical(self, query: LexicalQuery, filters: SearchFilters) -> list[LegHit]:
        """Lexical leg over current-version chunks, best first."""

    async def vector(self, query: VectorQuery, filters: SearchFilters) -> list[LegHit]:
        """Vector leg (cosine similarity scores) over current-version chunks, best first."""

    async def hits(self, chunk_ids: Sequence[ChunkId]) -> list[Hit]:
        """Display data for chunks (first current-version membership), in input order."""

    async def similarities(
        self, chunk_ids: Sequence[ChunkId], vector: Sequence[float]
    ) -> dict[ChunkId, float]:
        """Cosine similarity of the query to each chunk's active-space vector."""


class SearchStore(Protocol):
    """Opens search snapshots."""

    def snapshot(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> AbstractAsyncContextManager[SearchSnapshot]:
        """Begin a snapshot; it commits when the context exits."""


class Clock(Protocol):
    """Injected time source."""

    def now(self) -> float:
        """Seconds since the epoch."""

    def monotonic(self) -> float:
        """Monotonic seconds for durations."""


class TokenCounterFactory(Protocol):
    """Returns the token counter of a manifest tokenizer id (`chunking.tokenizer`)."""

    def for_tokenizer(self, tokenizer_id: str) -> TokenCounter:
        """Counter for `tokenizer_id`; raises UnavailableError when it cannot be loaded."""


class CollectionRepository(Protocol):
    """Collections of a tenant; other tenants' collections look absent."""

    async def create(self, tenant: TenantId, new: NewCollection) -> CollectionRecord:
        """Insert a collection; ConflictError when the name is taken."""

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, *, include_deleting: bool = False
    ) -> CollectionRecord:
        """One collection; `deleting` ones are absent unless asked for."""

    async def list_page(
        self, tenant: TenantId, after: CollectionId | None, limit: int
    ) -> list[CollectionRecord]:
        """Active collections ordered by id (keyset pagination)."""

    async def count(self, tenant: TenantId) -> int:
        """Number of collections (quota `max_collections`)."""

    async def total_chunks(self, tenant: TenantId) -> int:
        """Sum of `chunk_count` over the tenant's collections (quota `max_chunks`)."""

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        settings: CollectionSettings,
        *,
        webhook: tuple[str | None, str | None] | None = None,
        jobs: Sequence[NewJob] = (),
    ) -> tuple[CollectionRecord, list[JobId]]:
        """Replace stored settings, optionally the webhook (url, secret), and enqueue `jobs`.

        One transaction: a settings change and the job it starts commit together.
        """

    async def set_target_space(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int | None
    ) -> CollectionRecord:
        """Start (space id) or cancel (None) a re-embed; progress is reset."""

    async def set_reembed_progress(
        self, tenant: TenantId, collection_id: CollectionId, done: int, total: int
    ) -> None:
        """Record re-embed progress `{target_space_id, done, total, state}`."""

    async def mark_deleting(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> list[JobId]:
        """Hide the collection from reads and enqueue `jobs` (its `collection_delete`), one tx."""

    async def delete_rows(self, tenant: TenantId, collection_id: CollectionId) -> None:
        """Delete the collection and everything in it."""

    async def set_language(
        self, tenant: TenantId, collection_id: CollectionId, language: Language
    ) -> None:
        """`fts_reconfig`: switch `language` and every chunk's FTS config atomically."""


class BlobStore(Protocol):
    """Stored originals (`document_blobs`)."""

    async def source(self, tenant: TenantId, version_id: VersionId) -> bytes:
        """Original bytes of a version; NotFoundError `original_not_retained` when removed."""

    async def delete_original(self, tenant: TenantId, version_id: VersionId) -> None:
        """Remove a version's blob (`ingest.retain_originals: false`)."""


class DocumentRepository(BlobStore, Protocol):
    """Documents, versions, elements and originals."""

    async def upload(self, tenant: TenantId, up: NewUpload) -> UploadResult:
        """Create or version a document, store the original and enqueue `ingest` (one tx)."""

    async def reversion(
        self, tenant: TenantId, document_id: DocumentId, chunking_hash: str
    ) -> UploadResult | None:
        """New version from the current one's stored elements (reindex), `ingest` enqueued.

        None when the latest version already has `chunking_hash` (processing or ready) or
        the document has no current version.
        """

    async def uploaded_bytes_today(self, tenant: TenantId) -> int:
        """Bytes uploaded today (quota `max_upload_bytes_per_day`)."""

    async def count(self, tenant: TenantId) -> int:
        """Number of documents (quota `max_documents`)."""

    async def find(
        self, tenant: TenantId, collection_id: CollectionId, external_id: str
    ) -> DocumentRecord | None:
        """The document with `external_id` in an active collection, if any."""

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> DocumentRecord:
        """One document of an active collection."""

    async def list_page(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        after: DocumentId | None,
        limit: int,
    ) -> list[DocumentRecord]:
        """Documents of a collection ordered by id (keyset pagination)."""

    async def versions(self, tenant: TenantId, document_id: DocumentId) -> list[VersionRecord]:
        """Every version of a document, oldest first."""

    async def version(self, tenant: TenantId, version_id: VersionId) -> VersionRecord:
        """One version by id."""

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> DocumentRecord:
        """PATCH metadata and/or the display title."""

    async def set_status(
        self,
        tenant: TenantId,
        document_id: DocumentId,
        status: DocumentStatus,
        error: dict[str, Any] | None = None,
        *,
        version_id: VersionId | None = None,
    ) -> None:
        """Set the visible status; with `version_id`, only while no newer version exists."""

    async def store_elements(
        self,
        tenant: TenantId,
        version_id: VersionId,
        elements: list[Element],
        pages: int | None = None,
    ) -> None:
        """Store parsed elements so re-chunking never needs the original."""

    async def elements(self, tenant: TenantId, version_id: VersionId) -> list[Element]:
        """Stored elements of a version; NotFoundError when not parsed yet."""

    async def fail_version(
        self, tenant: TenantId, version_id: VersionId, error: dict[str, Any]
    ) -> None:
        """Mark a version failed; the document shows the error unless a newer version exists."""

    async def original(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        version: int | None = None,
    ) -> tuple[str, bytes]:
        """(media type, original bytes) of the current or a given version."""

    async def delete(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> None:
        """Delete a document with everything that belongs to it."""


class IndexRepository(Protocol):
    """Write side of retrieval: chunks, memberships, vectors, swap, switch and GC."""

    async def existing_chunks(
        self, tenant: TenantId, document_id: DocumentId, hashes: Sequence[str]
    ) -> dict[str, ChunkId]:
        """Chunk ids of the document already stored for these content hashes."""

    async def insert_chunks(self, tenant: TenantId, batch: ChunkBatch) -> dict[str, ChunkId]:
        """Insert chunks (one per distinct hash, existing ones kept); returns hash -> id."""

    async def insert_vectors(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        space: EmbeddingSpace,
        vectors: Sequence[tuple[ChunkId, Sequence[float]]],
    ) -> None:
        """Store L2-normalised vectors of `space` (existing rows kept)."""

    async def missing_vectors(
        self, tenant: TenantId, collection_id: CollectionId, chunk_ids: Sequence[ChunkId]
    ) -> dict[int, list[ChunkId]]:
        """Per needed space (active, and target while re-embedding): chunks lacking a vector."""

    async def current_chunk_count(self, tenant: TenantId, document_id: DocumentId) -> int:
        """Distinct chunks of the document's current version (0 without one)."""

    async def swap(  # noqa: PLR0913 - mirrors the swap transaction's inputs
        self,
        tenant: TenantId,
        document_id: DocumentId,
        version_id: VersionId,
        memberships: Sequence[Membership],
        *,
        jobs: Sequence[NewJob] = (),
        on_current: Sequence[NewJob] = (),
    ) -> SwapResult:
        """Version swap in one transaction; `on_current` jobs only when it becomes current.

        ConflictError `vectors_missing` when a needed space lacks a vector.
        """

    async def gc_document(
        self, tenant: TenantId, document_id: DocumentId, *, delete_superseded_originals: bool
    ) -> GcStats:
        """Remove superseded/failed memberships, unreferenced chunks and their vectors."""

    async def gc_space(self, tenant: TenantId, collection_id: CollectionId, space_id: int) -> int:
        """Delete a collection's vectors of a space that is neither active nor target."""

    async def reembed_batch(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int, limit: int
    ) -> list[tuple[ChunkId, str]]:
        """Current-version chunks lacking a vector of `space_id` (id, embedded text)."""

    async def reembed_counts(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> tuple[int, int]:
        """(current-version chunks with a `space_id` vector, all current-version chunks)."""

    async def switch_space(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> bool:
        """Space switch: active := target when every current-version chunk has a target vector."""


class JobQueue(Protocol):
    """Enqueues and reads jobs of a tenant."""

    async def enqueue(self, tenant: TenantId, job: NewJob) -> JobId:
        """Insert a queued job."""

    async def get(self, tenant: TenantId, job_id: JobId) -> JobRecord:
        """One job of the tenant; NotFoundError for other tenants' ids."""

    async def active_for_collection(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> list[JobRecord]:
        """Queued or running jobs whose args name the collection."""


class JobRun(Protocol):
    """What a job handler receives from the worker."""

    @property
    def job(self) -> ClaimedJob:
        """The claimed job."""

    async def progress(self, done: int, total: int) -> None:
        """Record `{done, total}` progress."""


class SpaceReader(Protocol):
    """Registered embedding spaces (global, read-only for footnote_app)."""

    async def list_spaces(self) -> list[EmbeddingSpace]:
        """All spaces ordered by id."""

    async def list_spaces_page(self, after: int | None, limit: int) -> list[EmbeddingSpace]:
        """Spaces ordered by id after `after` (keyset pagination)."""

    async def get(self, name: str) -> EmbeddingSpace:
        """A space by name; NotFoundError when absent."""

    async def get_by_id(self, space_id: int) -> EmbeddingSpace:
        """A space by id; NotFoundError when absent."""


class SpaceRegistry(SpaceReader, Protocol):
    """Registers and deletes spaces (admin pool)."""

    async def register(self, spec: SpaceSpec) -> EmbeddingSpace:
        """Idempotent by name; only a non-null threshold updates an existing row."""

    async def delete(self, name: str) -> None:
        """Drop the space; ConflictError `space_in_use` while a collection uses it."""


class TenantReader(Protocol):
    """Reads a tenant (admin pool; footnote_app has no privileges on `tenants`)."""

    async def get_tenant(self, tenant: TenantId) -> Tenant:
        """One tenant; NotFoundError when absent."""


class TenantAdmin(TenantReader, Protocol):
    """Tenant and key administration over the admin pool."""

    async def create_tenant(self, name: str, quotas: Quotas) -> Tenant:
        """Insert a tenant."""

    async def list_tenants(self, after: TenantId | None, limit: int) -> list[Tenant]:
        """Tenants ordered by id (keyset pagination)."""

    async def update_tenant(
        self, tenant: TenantId, *, name: str | None = None, quotas: Quotas | None = None
    ) -> Tenant:
        """Change the name and/or quotas."""

    async def mark_deleting(self, tenant: TenantId, jobs: Sequence[NewJob] = ()) -> list[JobId]:
        """Set status `deleting`, revoke every key and enqueue `jobs` (`tenant_delete`), one tx."""

    async def delete_tenant_data(self, tenant: TenantId) -> None:
        """Remove every row of the tenant (the `tenant_delete` job)."""

    async def usage(self, tenant: TenantId) -> TenantUsage:
        """Counts and stored bytes of a tenant."""

    async def create_key(
        self, tenant: TenantId, name: str, expires_at: datetime | None = None
    ) -> IssuedKey:
        """Issue a key; the plaintext is returned once."""

    async def list_keys(
        self, tenant: TenantId, after: UUID | None, limit: int
    ) -> list[ApiKeyRecord]:
        """Keys of a tenant (no secrets) ordered by id after `after` (keyset pagination)."""

    async def revoke_key(self, tenant: TenantId, key_id: UUID) -> None:
        """Revoke a key of this tenant (idempotent)."""

    async def rotate_key(self, tenant: TenantId, key_id: UUID, grace: timedelta) -> IssuedKey:
        """Issue a replacement; the old key stays valid for `grace`."""


class KeyResolver(Protocol):
    """Resolves tenant API keys (footnote_app via `footnote_resolve_key`)."""

    async def resolve(self, key: str) -> tuple[TenantId, UUID]:
        """(tenant id, key id); UnauthenticatedError for unknown, revoked or expired keys."""


class WebhookDeliveryLog(Protocol):
    """Webhook delivery log."""

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int:
        """Attempts recorded so far for a delivery (0 when it has no row yet)."""

    async def record_attempt(self, tenant: TenantId, attempt: AttemptRecord) -> None:
        """Insert the delivery on its first attempt, else bump `attempts` and the outcome."""

    async def list_for_collection(
        self, tenant: TenantId, collection_id: CollectionId, limit: int = 50
    ) -> list[DeliveryRecord]:
        """Newest deliveries of a collection first."""


class UploadInspector(Protocol):
    """Synchronous upload checks: content sniffing and pre-parse limits."""

    async def inspect(self, data: bytes, filename: str) -> InspectedFile:
        """Media type from content (never the client header) and page count.

        Raises DomainError `unsupported_media_type`, `pdf_encrypted`, `too_many_pages`,
        `archive_too_large` or `parse_failed`.
        """


class HostResolver(Protocol):
    """Resolves a webhook host to IP addresses (checked against the SSRF policy)."""

    async def resolve(self, host: str, port: int) -> Sequence[str]:
        """Addresses of `host`; an empty list when it does not resolve."""


class ChunkReader(Protocol):
    """Reads one chunk of a current version."""

    async def chunk(
        self, tenant: TenantId, collection_id: CollectionId, chunk_id: ChunkId
    ) -> ChunkDetail:
        """The chunk; NotFoundError `chunk_not_found` when no current version contains it."""


class TenantExporter(Protocol):
    """Streams a tenant's data (admin pool)."""

    def export(
        self, tenant: TenantId, *, include_vectors: bool, include_originals: bool
    ) -> AsyncIterator[ExportRecord]:
        """Records in dependency order; NotFoundError before the first record when absent."""
