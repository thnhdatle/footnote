# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database, Pool
from footnote.adapters.postgres.spaces import PgSpaceReader
from footnote.domain.documents import ChunkBatch, DocumentStatus, Membership, NewCollection
from footnote.domain.errors import NotFoundError
from footnote.domain.jobs import NewJob
from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.webhooks import WebhookEvent, WebhookSettings
from tests.integration.world import (
    World,
    drafts_for,
    hash_vector,
    index_version,
    seed,
    upload_of,
)

pytestmark = pytest.mark.integration


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def a(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, _name("a"), ["apple pie recipe", "banana bread recipe"])


@pytest_asyncio.fixture
async def b(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, _name("b"), ["cherry tart"])


@pytest.fixture(params=[True, False], ids=["predicates", "rls-only"])
def b_db(request: pytest.FixtureRequest, app_pool: Pool) -> Database:
    return Database(app_pool, app_tenant_predicates=bool(request.param))


async def _external_id(docs: PgDocuments, a: World) -> str:
    return (await docs.get(a.tenant, a.collection.id, a.document_id)).external_id


async def test_space_get_by_id(app_db: Database, a: World) -> None:
    reader = PgSpaceReader(app_db)
    assert await reader.get_by_id(a.space.id) == a.space
    with pytest.raises(NotFoundError):
        await reader.get_by_id(2_000_000_000)


async def test_total_chunks_is_tenant_scoped(app_db: Database, a: World, b: World) -> None:
    repo = PgCollections(app_db)
    assert await repo.total_chunks(a.tenant) == 2
    assert await repo.total_chunks(b.tenant) == 1
    assert await repo.total_chunks(uuid.uuid4()) == 0  # type: ignore[arg-type]


async def test_collection_webhook_events_round_trip(app_db: Database, a: World) -> None:
    repo = PgCollections(app_db)
    settings = CollectionSettings(
        embedding_space=a.space.name,
        webhook=WebhookSettings(events=(WebhookEvent.DOCUMENT_FAILED,)),
    )
    made = await repo.create(a.tenant, NewCollection(_name("hooks"), settings, a.space.id))
    assert made.settings.webhook.events == (WebhookEvent.DOCUMENT_FAILED,)
    assert (await repo.get(a.tenant, made.id)).settings.webhook == settings.webhook
    assert (await repo.get(a.tenant, a.collection.id)).settings.webhook == WebhookSettings()
    updated, _ = await repo.update(
        a.tenant, made.id, CollectionSettings(embedding_space=a.space.name)
    )
    assert updated.settings.webhook == WebhookSettings()


async def test_current_chunk_count_and_on_current_jobs(app_db: Database, a: World) -> None:
    docs, index, jobs = PgDocuments(app_db), PgIndex(app_db), PgJobs(app_db)
    assert await index.current_chunk_count(a.tenant, a.document_id) == 2
    fresh = await docs.upload(a.tenant, upload_of(a.collection, b"fresh", external_id="fresh"))
    assert await index.current_chunk_count(a.tenant, fresh.document_id) == 0

    ext = await _external_id(docs, a)
    older = await docs.upload(a.tenant, upload_of(a.collection, b"older", external_id=ext))
    newer_doc, _, _, _ = await index_version(
        app_db, a.collection, a.space, upload_of(a.collection, b"newer", external_id=ext), ["fig"]
    )
    assert older.version_id is not None
    assert await index.current_chunk_count(a.tenant, newer_doc) == 1

    drafts = drafts_for("Doc", ["grape juice"])
    ids = await index.insert_chunks(
        a.tenant, ChunkBatch(a.collection.id, a.document_id, "Doc", Language.SIMPLE, drafts)
    )
    chunk = ids[drafts[0].content_hash]
    await index.insert_vectors(a.tenant, a.collection.id, a.space, [(chunk, hash_vector("g"))])
    late = await index.swap(
        a.tenant,
        a.document_id,
        older.version_id,
        [Membership(older.version_id, 0, chunk, 1, 1)],
        jobs=[NewJob("gc", {"x": "always"})],
        on_current=[NewJob("webhook", {"x": "current"})],
    )
    assert not late.current
    assert [(await jobs.get(a.tenant, j)).args["x"] for j in late.job_ids] == ["always"]
    assert await index.current_chunk_count(a.tenant, a.document_id) == 1

    again = await docs.upload(a.tenant, upload_of(a.collection, b"third", external_id=ext))
    assert again.version_id is not None
    won = await index.swap(
        a.tenant,
        a.document_id,
        again.version_id,
        [Membership(again.version_id, 0, chunk, 1, 1)],
        jobs=[NewJob("gc", {"x": "always"})],
        on_current=[NewJob("webhook", {"x": "current"})],
    )
    assert won.current
    assert [(await jobs.get(a.tenant, j)).args["x"] for j in won.job_ids] == ["always", "current"]


async def test_set_status_guard(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    ext = await _external_id(docs, a)
    newer = await docs.upload(a.tenant, upload_of(a.collection, b"v2", external_id=ext))
    assert newer.version_id is not None

    async def status() -> DocumentStatus:
        return (await docs.get(a.tenant, a.collection.id, a.document_id)).status

    assert await status() is DocumentStatus.QUEUED
    await docs.set_status(a.tenant, a.document_id, DocumentStatus.PARSING, version_id=a.version_id)
    assert await status() is DocumentStatus.QUEUED
    await docs.set_status(
        a.tenant, a.document_id, DocumentStatus.PARSING, version_id=newer.version_id
    )
    assert await status() is DocumentStatus.PARSING
    await docs.set_status(a.tenant, a.document_id, DocumentStatus.EMBEDDING)
    assert await status() is DocumentStatus.EMBEDDING
    with pytest.raises(NotFoundError):
        await docs.set_status(a.tenant, uuid.uuid4(), DocumentStatus.PARSING)  # type: ignore[arg-type]
    await docs.set_status(
        a.tenant,
        uuid.uuid4(),  # type: ignore[arg-type]
        DocumentStatus.PARSING,
        version_id=newer.version_id,
    )


async def test_fail_version_guard(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    ext = await _external_id(docs, a)
    newer = await docs.upload(a.tenant, upload_of(a.collection, b"v2", external_id=ext))
    assert newer.version_id is not None
    error = {"code": "parse_failed", "message": "x"}
    await docs.fail_version(a.tenant, a.version_id, error)
    assert (await docs.version(a.tenant, a.version_id)).status.value == "failed"
    doc = await docs.get(a.tenant, a.collection.id, a.document_id)
    assert (doc.status, doc.error) == (DocumentStatus.QUEUED, None)
    await docs.fail_version(a.tenant, newer.version_id, error)
    doc = await docs.get(a.tenant, a.collection.id, a.document_id)
    assert (doc.status, doc.error) == (DocumentStatus.FAILED, error)


async def test_reversion(app_db: Database, a: World) -> None:
    docs, jobs = PgDocuments(app_db), PgJobs(app_db)
    result = await docs.reversion(a.tenant, a.document_id, "other-hash")
    assert result is not None
    assert (result.unchanged, result.version) == (False, 2)
    assert result.version_id is not None
    assert result.job_id is not None
    old = await docs.version(a.tenant, a.version_id)
    new = await docs.version(a.tenant, result.version_id)
    assert new.chunking_hash == "other-hash"
    assert new.status.value == "processing"
    assert (new.source_sha256, new.media_type, new.bytes, new.indexed_title) == (
        old.source_sha256,
        old.media_type,
        old.bytes,
        old.indexed_title,
    )
    assert await docs.elements(a.tenant, result.version_id) == await docs.elements(
        a.tenant, a.version_id
    )
    first = await docs.original(a.tenant, a.collection.id, a.document_id, 1)
    assert await docs.original(a.tenant, a.collection.id, a.document_id, 2) == first
    job = await jobs.get(a.tenant, result.job_id)
    assert job.kind == "ingest"
    assert job.args == {
        "collection_id": str(a.collection.id),
        "document_id": str(a.document_id),
        "version_id": str(result.version_id),
    }
    doc = await docs.get(a.tenant, a.collection.id, a.document_id)
    assert doc.status is DocumentStatus.QUEUED
    assert doc.current_version_id == a.version_id
    assert await docs.reversion(a.tenant, a.document_id, "other-hash") is None
    assert await docs.reversion(a.tenant, a.document_id, "third-hash") is None
    await docs.fail_version(a.tenant, result.version_id, {"code": "x", "message": "y"})
    retry = await docs.reversion(a.tenant, a.document_id, "other-hash")
    assert retry is not None
    assert retry.version == 3


async def test_reversion_without_current_version(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    fresh = await docs.upload(a.tenant, upload_of(a.collection, b"fresh", external_id="fresh"))
    assert await docs.reversion(a.tenant, fresh.document_id, "h") is None
    assert len(await docs.versions(a.tenant, fresh.document_id)) == 1


async def test_reversion_skips_ready_version_with_same_hash(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    same = (await docs.version(a.tenant, a.version_id)).chunking_hash
    assert await docs.reversion(a.tenant, a.document_id, same) is None
    assert len(await docs.versions(a.tenant, a.document_id)) == 1


async def test_reversion_copies_nothing_when_original_was_removed(
    app_db: Database, a: World
) -> None:
    docs = PgDocuments(app_db)
    await docs.delete_original(a.tenant, a.version_id)
    result = await docs.reversion(a.tenant, a.document_id, "other-hash")
    assert result is not None
    assert result.version_id is not None
    assert not (await docs.version(a.tenant, result.version_id)).has_original


async def test_ingest_methods_cross_tenant(
    b_db: Database, app_db: Database, a: World, b: World
) -> None:
    docs, index, repo = PgDocuments(b_db), PgIndex(b_db), PgCollections(b_db)
    assert await repo.total_chunks(b.tenant) == 1
    assert await index.current_chunk_count(b.tenant, a.document_id) == 0
    for call in (
        docs.reversion(b.tenant, a.document_id, "other-hash"),
        docs.set_status(b.tenant, a.document_id, DocumentStatus.FAILED),
        docs.fail_version(b.tenant, a.version_id, {"code": "x", "message": "y"}),
        index.swap(
            b.tenant,
            a.document_id,
            a.version_id,
            [],
            on_current=[NewJob("gc", {})],
        ),
    ):
        with pytest.raises(NotFoundError):
            await call
    await docs.set_status(b.tenant, a.document_id, DocumentStatus.FAILED, version_id=a.version_id)
    own = PgDocuments(app_db)
    doc = await own.get(a.tenant, a.collection.id, a.document_id)
    assert doc.status is DocumentStatus.READY
    assert (await own.version(a.tenant, a.version_id)).status.value == "ready"
    assert len(await own.versions(a.tenant, a.document_id)) == 1
    assert await PgIndex(app_db).current_chunk_count(a.tenant, a.document_id) == 2
