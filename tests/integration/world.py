# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID

import psycopg

from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.session import Database
from footnote.adapters.postgres.spaces import PgSpaceReader
from footnote.adapters.postgres.tenants import PgTenantAdmin
from footnote.domain.chunking import content_hash, embedded_text
from footnote.domain.documents import (
    ChunkBatch,
    ChunkDraft,
    ChunkId,
    CollectionRecord,
    DocumentId,
    Element,
    ElementKind,
    JobId,
    Membership,
    NewCollection,
    NewUpload,
    VersionId,
)
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.jobs import NewJob
from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.tenancy import Quotas, TenantId

SPACE_NAME = "it-fake-8"
DIMS = 8


def hash_vector(text: str, dims: int = DIMS) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(b - 127.5) / 127.5 for b in digest[:dims]]


def drafts_for(
    title: str, texts: Sequence[str], path: tuple[str, ...] = ("Intro",)
) -> list[ChunkDraft]:
    out = []
    for n, text in enumerate(texts):
        emb = embedded_text(title, path, text)
        out.append(
            ChunkDraft(
                ordinal=n,
                text=text,
                heading_path=path,
                embedded_text=emb,
                content_hash=content_hash(emb),
                token_count=len(text.split()),
                page_from=n + 1,
                page_to=n + 1,
            )
        )
    return out


@dataclass
class World:
    tenant: TenantId
    key: str
    key_id: UUID
    collection: CollectionRecord
    space: EmbeddingSpace
    document_id: DocumentId
    version_id: VersionId
    ingest_job: JobId
    chunk_ids: list[ChunkId] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


async def index_version(
    app_db: Database,
    collection: CollectionRecord,
    space: EmbeddingSpace,
    upload: NewUpload,
    texts: Sequence[str],
) -> tuple[DocumentId, VersionId, JobId, list[ChunkId]]:
    world_tenant = collection.tenant_id
    documents = PgDocuments(app_db)
    index = PgIndex(app_db)
    result = await documents.upload(world_tenant, upload)
    assert result.version_id is not None
    assert result.job_id is not None
    elements = [Element(ElementKind.PARAGRAPH, t, n) for n, t in enumerate(texts)]
    await documents.store_elements(world_tenant, result.version_id, elements, pages=len(texts))
    drafts = drafts_for(upload.title, texts)
    ids = await index.insert_chunks(
        world_tenant,
        ChunkBatch(collection.id, result.document_id, upload.title, Language.SIMPLE, drafts),
    )
    await index.insert_vectors(
        world_tenant,
        collection.id,
        space,
        [(ids[d.content_hash], hash_vector(d.text)) for d in drafts],
    )
    memberships = [
        Membership(result.version_id, d.ordinal, ids[d.content_hash], d.page_from, d.page_to)
        for d in drafts
    ]
    await index.swap(
        world_tenant,
        result.document_id,
        result.version_id,
        memberships,
        jobs=[NewJob("gc", {"document_id": str(result.document_id)}, timedelta(minutes=10))],
    )
    return result.document_id, result.version_id, result.job_id, [m.chunk_id for m in memberships]


def upload_of(
    collection: CollectionRecord,
    data: bytes,
    *,
    external_id: str | None = None,
    title: str = "Doc",
    metadata: dict[str, object] | None = None,
) -> NewUpload:
    return NewUpload(
        collection_id=collection.id,
        external_id=external_id,
        title=title,
        metadata=dict(metadata if metadata is not None else {"lang": "en"}),
        media_type="text/plain",
        source_sha256=hashlib.sha256(data).digest(),
        data=data,
        chunking_hash=CollectionSettings().chunking.settings_hash(),
    )


async def seed(
    admin_db: Database,
    app_db: Database,
    name: str,
    texts: Sequence[str],
    space_name: str = SPACE_NAME,
) -> World:
    admin = PgTenantAdmin(admin_db)
    tenant = await admin.create_tenant(name, Quotas())
    issued = await admin.create_key(tenant.id, "default")
    space = await PgSpaceReader(app_db).get(space_name)
    collection = await PgCollections(app_db).create(
        tenant.id,
        NewCollection(
            name="docs",
            settings=CollectionSettings(embedding_space=space.name),
            space_id=space.id,
            webhook_url="https://example.com/hook",
            webhook_secret="s3cret",
        ),
    )
    data = "\n\n".join(texts).encode()
    doc, version, job, chunk_ids = await index_version(
        app_db, collection, space, upload_of(collection, data), texts
    )
    async with app_db.tenant_tx(tenant.id) as conn:
        await conn.execute(
            "INSERT INTO footnote.webhook_deliveries (tenant_id, id, collection_id, event, url) "
            "VALUES (%s, gen_random_uuid(), %s, 'document.ready', 'https://example.com/hook')",
            (tenant.id, collection.id),
        )
    return World(
        tenant=tenant.id,
        key=issued.key,
        key_id=issued.record.id,
        collection=collection,
        space=space,
        document_id=doc,
        version_id=version,
        ingest_job=job,
        chunk_ids=chunk_ids,
        texts=list(texts),
    )


async def app_connection(dsn: str) -> psycopg.AsyncConnection[tuple[object, ...]]:
    return await psycopg.AsyncConnection.connect(dsn, autocommit=True)
