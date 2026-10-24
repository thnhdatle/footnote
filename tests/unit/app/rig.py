# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from footnote.app.gc import GcService
from footnote.app.ingest import IngestConfig, IngestDeps, IngestService, IngestTarget
from footnote.app.jobs import JobHandlers, JobServices
from footnote.app.reembed import ReembedConfig, ReembedService
from footnote.app.reindex import ReindexService
from footnote.app.webhooks import WebhookNotifier
from footnote.domain.documents import (
    CollectionId,
    CollectionRecord,
    DocumentId,
    NewCollection,
    NewUpload,
    UploadResult,
    VersionId,
)
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.ports import Embedder
from footnote.domain.settings import ChunkingSettings, ChunkingStrategyName, CollectionSettings
from footnote.domain.tenancy import Quotas, TenantId
from footnote.domain.webhooks import WebhookSettings

from .fakes import (
    DIMS,
    FakeClock,
    FakeCollections,
    FakeDb,
    FakeDocuments,
    FakeIndex,
    FakeJobs,
    FakeParser,
    FakeSpaces,
    FakeTenants,
    FakeTokenizers,
    hash_vector,
)

SPACE_A = EmbeddingSpace(1, "space-a", "fake", "model-a", DIMS)
SPACE_B = EmbeddingSpace(2, "space-b", "fake", "model-b", DIMS)
GC_GRACE = timedelta(minutes=10)
CHUNKING = ChunkingSettings(
    strategy=ChunkingStrategyName.RECURSIVE, max_tokens=32, overlap_tokens=0, min_tokens=0
)


def paragraph(word: str) -> str:
    return " ".join(f"{word}{n}" for n in range(20))


class SpaceEmbedder:
    def __init__(self, space: EmbeddingSpace, owner: RecordingEmbedders) -> None:
        self.space = space
        self.owner = owner

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.owner.calls.append((self.space.id, list(texts)))
        if self.owner.before_embed is not None:
            await self.owner.before_embed(self.space.id)
        if self.owner.error is not None:
            raise self.owner.error
        vectors = [hash_vector(f"{self.space.id}:{t}") for t in texts]
        return vectors[: self.owner.limit] if self.owner.limit is not None else vectors

    async def embed_query(self, text: str) -> list[float]:
        return hash_vector(f"{self.space.id}:{text}")


@dataclass
class RecordingEmbedders:
    calls: list[tuple[int, list[str]]] = field(default_factory=list)
    error: Exception | None = None
    limit: int | None = None
    before_embed: Callable[[int], object] | None = None

    def for_space(self, space: EmbeddingSpace) -> Embedder:
        return SpaceEmbedder(space, self)

    def texts(self, space_id: int | None = None) -> list[str]:
        return [t for s, ts in self.calls for t in ts if space_id is None or s == space_id]


@dataclass
class Rig:
    db: FakeDb
    tenant: TenantId
    collection: CollectionRecord
    collections: FakeCollections
    documents: FakeDocuments
    index: FakeIndex
    jobs: FakeJobs
    spaces: FakeSpaces
    tenants: FakeTenants
    parser: FakeParser
    embedders: RecordingEmbedders
    ingest: IngestService
    reindex: ReindexService
    reembed: ReembedService
    gc: GcService
    handlers: JobHandlers
    stages: list[str]

    async def upload(
        self, texts: Sequence[str], *, external_id: str = "doc", title: str = "Doc"
    ) -> UploadResult:
        data = "\n\n".join(texts).encode()
        return await self.documents.upload(
            self.tenant,
            NewUpload(
                self.collection.id,
                external_id,
                title,
                {"k": "v"},
                "text/plain",
                hashlib.sha256(data).digest(),
                data,
                self.collection.settings.chunking.settings_hash(),
            ),
        )

    def target(self, result: UploadResult) -> IngestTarget:
        assert result.version_id is not None
        return IngestTarget(self.collection.id, result.document_id, result.version_id)

    async def index_texts(self, texts: Sequence[str], **kw: object) -> UploadResult:
        result = await self.upload(texts, **kw)  # type: ignore[arg-type]
        await self.ingest.ingest(self.tenant, self.target(result))
        return result

    def current_texts(self) -> set[str]:
        chunk_ids = self.db.current_chunks(self.tenant, self.collection.id)
        return {
            self.db.chunks[(self.tenant, c)].embedded_text.rsplit("\n\n", 1)[1] for c in chunk_ids
        }


def build(  # noqa: PLR0913 - test rig knobs
    *,
    webhook: bool = True,
    events: WebhookSettings | None = None,
    retain_originals: bool = True,
    quotas: Quotas | None = None,
    parser_error: Exception | None = None,
    reembed_config: ReembedConfig | None = None,
) -> Rig:
    db = FakeDb()
    db.add_space(SPACE_A)
    db.add_space(SPACE_B)
    tenant = db.add_tenant(quotas)
    collections = FakeCollections(db)
    documents = FakeDocuments(db)
    index = FakeIndex(db)
    jobs = FakeJobs(db)
    spaces = FakeSpaces(db)
    tenants = FakeTenants(db)
    parser = FakeParser(parser_error)
    embedders = RecordingEmbedders()
    clock = FakeClock()
    stages: list[str] = []
    settings = CollectionSettings(
        chunking=CHUNKING, embedding_space=SPACE_A.name, webhook=events or WebhookSettings()
    )
    record = db_create(collections, tenant, settings, webhook=webhook)
    notifier = WebhookNotifier(jobs, clock)
    ingest = IngestService(
        IngestDeps(
            collections,
            documents,
            index,
            spaces,
            parser,
            embedders,
            FakeTokenizers(),
            notifier,
            clock,
            tenants,
        ),
        IngestConfig(gc_grace=GC_GRACE, retain_originals=retain_originals, batch_size=2),
        on_stage=lambda stage, _seconds: stages.append(stage),
    )
    reindex = ReindexService(collections, documents, page_size=2)
    reembed = ReembedService(
        collections,
        index,
        spaces,
        embedders,
        jobs,
        config=reembed_config or ReembedConfig(batch_size=2, gc_grace=GC_GRACE),
    )
    gc = GcService(index, retain_originals=retain_originals)
    handlers = JobHandlers(
        JobServices(jobs, collections, ingest, reindex, reembed, gc, tenants, max_attempts=3)  # type: ignore[arg-type]
    )
    return Rig(
        db,
        tenant,
        record,
        collections,
        documents,
        index,
        jobs,
        spaces,
        tenants,
        parser,
        embedders,
        ingest,
        reindex,
        reembed,
        gc,
        handlers,
        stages,
    )


def db_create(
    collections: FakeCollections, tenant: TenantId, settings: CollectionSettings, *, webhook: bool
) -> CollectionRecord:
    return collections.create_now(
        tenant,
        NewCollection(
            "docs",
            settings,
            SPACE_A.id,
            webhook_url="https://example.com/hook" if webhook else None,
            webhook_secret="s3cret" if webhook else None,
        ),
    )


__all__ = [
    "CHUNKING",
    "GC_GRACE",
    "SPACE_A",
    "SPACE_B",
    "CollectionId",
    "DocumentId",
    "Rig",
    "VersionId",
    "build",
    "paragraph",
]
