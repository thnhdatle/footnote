# SPDX-License-Identifier: Apache-2.0
"""Repositories against real Postgres; every method also runs cross-tenant.

The cross-tenant side runs twice: with the defence-in-depth predicates and with them
removed (the tenant-isolation fault mode), so RLS alone must hold.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta

import psycopg
import pytest
import pytest_asyncio

from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database, Pool
from footnote.adapters.postgres.spaces import HnswParams, PgSpaceAdmin
from footnote.adapters.postgres.tenants import PgKeyResolver, PgTenantAdmin
from footnote.domain.documents import (
    ChunkBatch,
    DocumentStatus,
    Element,
    ElementKind,
    Membership,
    NewCollection,
)
from footnote.domain.embedding import SpaceSpec
from footnote.domain.errors import ConflictError, NotFoundError, UnauthenticatedError
from footnote.domain.jobs import NewJob
from footnote.domain.settings import CollectionSettings, Language, RetrievalSettings
from footnote.domain.tenancy import Quotas
from tests.integration.conftest import PgInfo
from tests.integration.world import (
    World,
    drafts_for,
    hash_vector,
    index_version,
    seed,
    upload_of,
)

pytestmark = pytest.mark.integration

ALL_TABLES = (
    "api_keys",
    "collections",
    "documents",
    "document_versions",
    "document_blobs",
    "chunks",
    "version_chunks",
    "chunk_vectors",
    "jobs",
    "webhook_deliveries",
    "tenant_usage",
)


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def a(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    """Tenant A with one indexed document."""
    yield await seed(admin_db, app_db, _name("a"), ["apple pie recipe", "banana bread recipe"])


@pytest_asyncio.fixture
async def b(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    """Tenant B (the intruder)."""
    yield await seed(admin_db, app_db, _name("b"), ["cherry tart"])


@pytest.fixture(params=[True, False], ids=["predicates", "rls-only"])
def b_db(request: pytest.FixtureRequest, app_pool: Pool) -> Database:
    """The intruder's database, with or without app-side tenant predicates."""
    return Database(app_pool, app_tenant_predicates=bool(request.param))


async def _admin_count(pg: PgInfo, table: str, tenant: uuid.UUID) -> int:
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            f"SELECT count(*) FROM footnote.{table} WHERE tenant_id = %s",  # noqa: S608 - constant
            (tenant,),
        )
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])


# --- tenants and keys (admin pool) -------------------------------------------------------


async def test_tenant_admin_crud_and_usage(admin_db: Database, a: World, b: World) -> None:
    admin = PgTenantAdmin(admin_db)
    tenant = await admin.get_tenant(a.tenant)
    assert tenant.quotas == Quotas()
    updated = await admin.update_tenant(a.tenant, quotas=Quotas(max_collections=3))
    assert updated.quotas.max_collections == 3
    assert updated.name == tenant.name
    renamed = await admin.update_tenant(a.tenant, name="renamed")
    assert (renamed.name, renamed.quotas.max_collections) == ("renamed", 3)
    listed = await admin.list_tenants(None, 1000)
    assert {a.tenant, b.tenant} <= {t.id for t in listed}
    after = await admin.list_tenants(min(a.tenant, b.tenant), 1000)
    assert min(a.tenant, b.tenant) not in {t.id for t in after}
    usage = await admin.usage(a.tenant)
    assert (usage.collections, usage.documents, usage.chunks, usage.vectors) == (1, 1, 2, 2)
    assert usage.stored_bytes > 0
    for missing in (admin.get_tenant, admin.usage, admin.mark_deleting):
        with pytest.raises(NotFoundError):
            await missing(uuid.uuid4())  # type: ignore[arg-type]
    with pytest.raises(NotFoundError):
        await admin.update_tenant(uuid.uuid4(), name="x")  # type: ignore[arg-type]


async def test_keys_are_tenant_scoped(
    admin_db: Database, app_db: Database, a: World, b: World
) -> None:
    admin = PgTenantAdmin(admin_db)
    assert [k.id for k in await admin.list_keys(a.tenant, None, 100)] == [a.key_id]
    assert a.key_id not in {k.id for k in await admin.list_keys(b.tenant, None, 100)}
    assert await admin.list_keys(a.tenant, a.key_id, 100) == []
    with pytest.raises(NotFoundError):
        await admin.revoke_key(b.tenant, a.key_id)
    with pytest.raises(NotFoundError):
        await admin.rotate_key(b.tenant, a.key_id, timedelta(hours=1))
    with pytest.raises(NotFoundError):
        await admin.create_key(uuid.uuid4(), "x")  # type: ignore[arg-type]
    assert (await PgKeyResolver(app_db).resolve(a.key))[0] == a.tenant
    key = (await admin.list_keys(a.tenant, None, 100))[0]
    assert key.prefix == a.key[:11]
    assert key.revoked_at is None


async def test_tenant_delete_removes_every_row_and_spares_others(
    pg: PgInfo, admin_db: Database, app_db: Database, a: World, b: World
) -> None:
    admin = PgTenantAdmin(admin_db)
    await PgJobs(app_db).enqueue(a.tenant, NewJob("tenant_delete"))
    await admin.mark_deleting(a.tenant)
    with pytest.raises(UnauthenticatedError):
        await PgKeyResolver(app_db).resolve(a.key)
    before_b = {t: await _admin_count(pg, t, b.tenant) for t in ALL_TABLES}
    await admin.delete_tenant_data(a.tenant)
    for table in ALL_TABLES:
        assert await _admin_count(pg, table, a.tenant) == 0, table
        assert await _admin_count(pg, table, b.tenant) == before_b[table], table
    with pytest.raises(NotFoundError):
        await admin.get_tenant(a.tenant)


# --- collections -----------------------------------------------------------------------


async def test_collections_crud(app_db: Database, a: World) -> None:
    repo = PgCollections(app_db)
    got = await repo.get(a.tenant, a.collection.id)
    assert (got.id, got.name, got.tenant_id) == (a.collection.id, "docs", a.tenant)
    assert got.settings == a.collection.settings
    assert got.settings.embedding_space == a.space.name
    assert got.chunk_count == 2
    with pytest.raises(ConflictError):
        await repo.create(a.tenant, NewCollection("docs", CollectionSettings(), a.space.id))
    second = await repo.create(
        a.tenant,
        NewCollection(
            "second",
            CollectionSettings(
                retrieval=RetrievalSettings(k=3, rerank=False), language=Language.ENGLISH
            ),
            a.space.id,
        ),
    )
    assert second.settings.retrieval.k == 3
    assert second.settings.language is Language.ENGLISH
    assert await repo.count(a.tenant) == 2
    page = await repo.list_page(a.tenant, None, 1)
    rest = await repo.list_page(a.tenant, page[0].id, 10)
    assert {c.id for c in page + rest} == {a.collection.id, second.id}
    changed = replace(second.settings, retrieval=RetrievalSettings(k=5))
    updated, _ = await repo.update(
        a.tenant, second.id, changed, webhook=("https://example.org/h", "secret2")
    )
    assert (updated.settings.retrieval.k, updated.webhook_url) == (5, "https://example.org/h")
    kept, _ = await repo.update(a.tenant, second.id, changed)
    assert kept.webhook_secret == "secret2"
    await repo.mark_deleting(a.tenant, second.id)
    with pytest.raises(NotFoundError):
        await repo.get(a.tenant, second.id)
    assert (await repo.get(a.tenant, second.id, include_deleting=True)).status == "deleting"
    assert second.id not in {c.id for c in await repo.list_page(a.tenant, None, 10)}
    await repo.delete_rows(a.tenant, second.id)
    with pytest.raises(NotFoundError):
        await repo.get(a.tenant, second.id, include_deleting=True)


async def test_collections_cross_tenant(
    b_db: Database, app_db: Database, a: World, b: World
) -> None:
    repo = PgCollections(b_db)
    cid = a.collection.id
    with pytest.raises(NotFoundError):
        await repo.get(b.tenant, cid, include_deleting=True)
    assert cid not in {c.id for c in await repo.list_page(b.tenant, None, 100)}
    assert await repo.count(b.tenant) == 1
    for call in (
        repo.update(b.tenant, cid, CollectionSettings()),
        repo.set_target_space(b.tenant, cid, None),
        repo.mark_deleting(b.tenant, cid),
        repo.set_language(b.tenant, cid, Language.ENGLISH),
    ):
        with pytest.raises(NotFoundError):
            await call
    await repo.set_reembed_progress(b.tenant, cid, 1, 2)
    await repo.delete_rows(b.tenant, cid)
    # Same name in another tenant is fine.
    other = await repo.create(b.tenant, NewCollection("second", CollectionSettings(), b.space.id))
    assert other.tenant_id == b.tenant
    survivor = await PgCollections(app_db).get(a.tenant, cid)
    assert (survivor.status, survivor.settings, survivor.chunk_count) == (
        "active",
        a.collection.settings,
        2,
    )


async def test_set_language_reconfigures_chunks(pg: PgInfo, app_db: Database, a: World) -> None:
    await PgCollections(app_db).set_language(a.tenant, a.collection.id, Language.ENGLISH)
    got = await PgCollections(app_db).get(a.tenant, a.collection.id)
    assert got.settings.language is Language.ENGLISH
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT DISTINCT fts_config::text, tsv @@ to_tsquery('english', 'recipes') "
            "FROM footnote.chunks WHERE collection_id = %s",
            (a.collection.id,),
        )
        assert await cur.fetchall() == [("english", True)]


# --- documents ---------------------------------------------------------------------------


async def test_upload_unchanged_and_new_versions(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    record = await docs.get(a.tenant, a.collection.id, a.document_id)
    assert (record.status, record.current_version) == (DocumentStatus.READY, 1)
    data = "\n\n".join(a.texts).encode()
    same = upload_of(a.collection, data, external_id=record.external_id, metadata={"v": 2})
    unchanged = await docs.upload(a.tenant, same)
    assert unchanged.unchanged
    assert unchanged.document_id == a.document_id
    assert unchanged.job_id is None
    assert (await docs.get(a.tenant, a.collection.id, a.document_id)).metadata == {"v": 2}

    changed = upload_of(a.collection, b"new text", external_id=record.external_id)
    second = await docs.upload(a.tenant, changed)
    assert (second.unchanged, second.version) == (False, 2)
    assert second.upload_bytes_today >= len(b"new text")
    assert await docs.uploaded_bytes_today(a.tenant) == second.upload_bytes_today
    retitled = await docs.upload(a.tenant, replace(changed, title="Other"))
    assert retitled.version == 3
    versions = await docs.versions(a.tenant, a.document_id)
    assert [v.version for v in versions] == [1, 2, 3]
    assert versions[1].source_sha256 == hashlib.sha256(b"new text").digest()
    assert all(v.has_original for v in versions)
    assert (await docs.version(a.tenant, versions[2].id)).indexed_title == "Other"
    assert await docs.source(a.tenant, versions[1].id) == b"new text"
    assert await docs.original(a.tenant, a.collection.id, a.document_id) == ("text/plain", data)
    assert (await docs.original(a.tenant, a.collection.id, a.document_id, 2))[1] == b"new text"
    fresh = await docs.upload(a.tenant, upload_of(a.collection, b"other doc"))
    assert fresh.version == 1
    assert fresh.document_id != a.document_id
    assert await docs.count(a.tenant) == 2
    page = await docs.list_page(a.tenant, a.collection.id, None, 10)
    assert {d.id for d in page} == {a.document_id, fresh.document_id}


async def test_document_lifecycle_methods(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    patched = await docs.update(
        a.tenant, a.collection.id, a.document_id, metadata={"k": "v"}, title="Shown"
    )
    assert (patched.metadata, patched.title) == ({"k": "v"}, "Shown")
    await docs.set_status(a.tenant, a.document_id, DocumentStatus.PARSING)
    assert (
        await docs.get(a.tenant, a.collection.id, a.document_id)
    ).status is DocumentStatus.PARSING
    elements = [Element(ElementKind.HEADING, "H", 0, level=1, heading_path=("H",), page=1)]
    await docs.store_elements(a.tenant, a.version_id, elements, pages=3)
    assert await docs.elements(a.tenant, a.version_id) == elements
    assert (await docs.version(a.tenant, a.version_id)).pages == 3
    await docs.fail_version(a.tenant, a.version_id, {"code": "parse_failed", "message": "x"})
    failed = await docs.get(a.tenant, a.collection.id, a.document_id)
    assert failed.status is DocumentStatus.FAILED
    assert failed.error == {"code": "parse_failed", "message": "x"}
    await docs.delete_original(a.tenant, a.version_id)
    with pytest.raises(NotFoundError) as gone:
        await docs.original(a.tenant, a.collection.id, a.document_id)
    assert gone.value.code == "original_not_retained"
    with pytest.raises(NotFoundError):
        await docs.source(a.tenant, a.version_id)
    fresh = await docs.upload(a.tenant, upload_of(a.collection, b"unparsed"))
    assert fresh.version_id is not None
    with pytest.raises(NotFoundError):
        await docs.elements(a.tenant, fresh.version_id)


async def test_document_delete_cascades_and_updates_chunk_count(
    pg: PgInfo, app_db: Database, a: World
) -> None:
    await PgDocuments(app_db).delete(a.tenant, a.collection.id, a.document_id)
    for table in (
        "documents",
        "document_versions",
        "document_blobs",
        "chunks",
        "version_chunks",
        "chunk_vectors",
    ):
        assert await _admin_count(pg, table, a.tenant) == 0, table
    assert (await PgCollections(app_db).get(a.tenant, a.collection.id)).chunk_count == 0
    with pytest.raises(NotFoundError):
        await PgDocuments(app_db).delete(a.tenant, a.collection.id, a.document_id)


async def test_documents_cross_tenant(b_db: Database, app_db: Database, a: World, b: World) -> None:
    docs = PgDocuments(b_db)
    cid, did, vid = a.collection.id, a.document_id, a.version_id
    with pytest.raises(NotFoundError):
        await docs.upload(b.tenant, upload_of(a.collection, b"intrusion"))
    for call in (
        docs.get(b.tenant, cid, did),
        docs.version(b.tenant, vid),
        docs.update(b.tenant, cid, did, title="x"),
        docs.set_status(b.tenant, did, DocumentStatus.FAILED),
        docs.store_elements(b.tenant, vid, []),
        docs.elements(b.tenant, vid),
        docs.fail_version(b.tenant, vid, {"code": "x"}),
        docs.original(b.tenant, cid, did),
        docs.source(b.tenant, vid),
        docs.delete(b.tenant, cid, did),
    ):
        with pytest.raises(NotFoundError):
            await call
    assert await docs.list_page(b.tenant, cid, None, 10) == []
    assert await docs.versions(b.tenant, did) == []
    assert await docs.count(b.tenant) == 1
    await docs.delete_original(b.tenant, vid)
    own = PgDocuments(app_db)
    assert (await own.get(a.tenant, cid, did)).status is DocumentStatus.READY
    assert await own.source(a.tenant, vid) == "\n\n".join(a.texts).encode()


# --- index: chunks, vectors, swap, gc, re-embed ------------------------------------------


async def test_incremental_version_swap_and_gc(pg: PgInfo, app_db: Database, a: World) -> None:
    index = PgIndex(app_db)
    docs = PgDocuments(app_db)
    record = await docs.get(a.tenant, a.collection.id, a.document_id)
    texts = [a.texts[0], "durian smoothie recipe", "elderberry jam"]
    hashes = [d.content_hash for d in drafts_for("Doc", texts)]
    existing = await index.existing_chunks(a.tenant, a.document_id, hashes)
    assert set(existing) == {hashes[0]}
    upload = upload_of(a.collection, b"v2", external_id=record.external_id)
    _, v2, _, chunk_ids = await index_version(app_db, a.collection, a.space, upload, texts)
    assert chunk_ids[0] == a.chunk_ids[0]  # reused chunk row
    col = await PgCollections(app_db).get(a.tenant, a.collection.id)
    assert col.chunk_count == 3
    versions = {v.version: v.status.value for v in await docs.versions(a.tenant, a.document_id)}
    assert versions == {1: "superseded", 2: "ready"}
    assert (await docs.get(a.tenant, a.collection.id, a.document_id)).current_version_id == v2

    stats = await index.gc_document(a.tenant, a.document_id, delete_superseded_originals=True)
    assert (stats.memberships, stats.chunks, stats.blobs) == (2, 1, 1)
    assert await _admin_count(pg, "chunks", a.tenant) == 3
    assert await _admin_count(pg, "chunk_vectors", a.tenant) == 3
    again = await index.gc_document(a.tenant, a.document_id, delete_superseded_originals=True)
    assert (again.memberships, again.chunks, again.blobs) == (0, 0, 0)


async def test_older_version_swapped_late_is_superseded(app_db: Database, a: World) -> None:
    docs = PgDocuments(app_db)
    index = PgIndex(app_db)
    ext = (await docs.get(a.tenant, a.collection.id, a.document_id)).external_id
    older = await docs.upload(a.tenant, upload_of(a.collection, b"older", external_id=ext))
    newer_doc, newer, _, _ = await index_version(
        app_db,
        a.collection,
        a.space,
        upload_of(a.collection, b"newer", external_id=ext),
        ["fig salad"],
    )
    assert older.version_id is not None
    drafts = drafts_for("Doc", ["grape juice"])
    ids = await index.insert_chunks(
        a.tenant, ChunkBatch(a.collection.id, a.document_id, "Doc", Language.SIMPLE, drafts)
    )
    chunk = ids[drafts[0].content_hash]
    with pytest.raises(ConflictError) as missing:
        await index.swap(
            a.tenant,
            a.document_id,
            older.version_id,
            [Membership(older.version_id, 0, chunk, 1, 1)],
        )
    assert missing.value.code == "vectors_missing"
    assert await index.missing_vectors(a.tenant, a.collection.id, [chunk]) == {a.space.id: [chunk]}
    await index.insert_vectors(a.tenant, a.collection.id, a.space, [(chunk, hash_vector("g"))])
    result = await index.swap(
        a.tenant, a.document_id, older.version_id, [Membership(older.version_id, 0, chunk, 1, 1)]
    )
    assert not result.current
    assert result.chunk_count == 1
    current = await docs.get(a.tenant, a.collection.id, newer_doc)
    assert current.current_version_id == newer
    assert (await docs.version(a.tenant, older.version_id)).status.value == "superseded"


async def test_vectors_are_normalised_and_dimension_checked(
    pg: PgInfo, app_db: Database, a: World
) -> None:
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT min(vector_norm(embedding)), max(vector_norm(embedding)) "
            "FROM footnote.chunk_vectors WHERE tenant_id = %s",
            (a.tenant,),
        )
        low, high = await cur.fetchone() or (0.0, 0.0)
    assert abs(low - 1.0) <= 1e-6
    assert abs(high - 1.0) <= 1e-6
    with pytest.raises(Exception, match="dimensions") as err:
        await PgIndex(app_db).insert_vectors(
            a.tenant, a.collection.id, a.space, [(a.chunk_ids[0], [1.0, 2.0])]
        )
    assert getattr(err.value, "code", None) == "embedding_dimension_mismatch"


async def test_reembed_backfill_and_switch(
    pg: PgInfo, admin_db: Database, app_db: Database, a: World
) -> None:
    index = PgIndex(app_db)
    collections = PgCollections(app_db)
    target = await PgSpaceAdmin(admin_db, HnswParams(), 1000).register(
        SpaceSpec(_name("target"), "fake", "other-model", 4)
    )
    assert not await index.switch_space(a.tenant, a.collection.id)
    started = await collections.set_target_space(a.tenant, a.collection.id, target.id)
    assert started.target_space_id == target.id
    assert started.settings.embedding_space == target.name
    assert await index.reembed_counts(a.tenant, a.collection.id, target.id) == (0, 2)
    missing = await index.missing_vectors(a.tenant, a.collection.id, a.chunk_ids)
    assert missing == {target.id: sorted(a.chunk_ids)}
    batch = await index.reembed_batch(a.tenant, a.collection.id, target.id, 1)
    assert len(batch) == 1
    await index.insert_vectors(
        a.tenant, a.collection.id, target, [(cid, hash_vector(t, 4)) for cid, t in batch]
    )
    await collections.set_reembed_progress(a.tenant, a.collection.id, 1, 2)
    assert (await collections.get(a.tenant, a.collection.id)).reembed == {
        "target_space_id": target.id,
        "done": 1,
        "total": 2,
        "state": "running",
    }
    assert not await index.switch_space(a.tenant, a.collection.id)
    rest = await index.reembed_batch(a.tenant, a.collection.id, target.id, 10)
    await index.insert_vectors(
        a.tenant, a.collection.id, target, [(cid, hash_vector(t, 4)) for cid, t in rest]
    )
    gc = NewJob("gc", {"space_id": a.space.id}, timedelta(minutes=10))
    assert await index.switch_space(a.tenant, a.collection.id, [gc])
    switched = await collections.get(a.tenant, a.collection.id)
    assert (switched.active_space_id, switched.target_space_id) == (target.id, None)
    assert await index.gc_space(a.tenant, a.collection.id, target.id) == 0  # active: kept
    assert await index.gc_space(a.tenant, a.collection.id, a.space.id) == 2
    assert await _admin_count(pg, "chunk_vectors", a.tenant) == 2


async def test_cancel_reembed_clears_target(app_db: Database, admin_db: Database, a: World) -> None:
    target = await PgSpaceAdmin(admin_db, HnswParams(), 1000).register(
        SpaceSpec(_name("cancel"), "fake", "m", 4)
    )
    collections = PgCollections(app_db)
    await collections.set_target_space(a.tenant, a.collection.id, target.id)
    cancelled = await collections.set_target_space(a.tenant, a.collection.id, None)
    assert (cancelled.target_space_id, cancelled.reembed) == (None, None)
    assert cancelled.settings.embedding_space == a.space.name


async def test_index_cross_tenant(b_db: Database, app_db: Database, a: World, b: World) -> None:
    index = PgIndex(b_db)
    cid, did = a.collection.id, a.document_id
    hashes = [d.content_hash for d in drafts_for("Doc", a.texts)]
    assert await index.existing_chunks(b.tenant, did, hashes) == {}
    with pytest.raises(NotFoundError):
        await index.insert_chunks(
            b.tenant,
            ChunkBatch(cid, did, "x", Language.SIMPLE, drafts_for("x", ["intrusion"])),
        )
    with pytest.raises(NotFoundError):
        await index.insert_vectors(b.tenant, cid, a.space, [(a.chunk_ids[0], hash_vector("x"))])
    assert await index.missing_vectors(b.tenant, cid, a.chunk_ids) == {}
    for call in (
        index.swap(b.tenant, did, a.version_id, []),
        index.gc_document(b.tenant, did, delete_superseded_originals=True),
        index.switch_space(b.tenant, cid),
    ):
        with pytest.raises(NotFoundError):
            await call
    assert await index.gc_space(b.tenant, cid, a.space.id) == 0
    assert await index.reembed_batch(b.tenant, cid, a.space.id, 10) == []
    assert await index.reembed_counts(b.tenant, cid, a.space.id) == (0, 0)
    own = PgIndex(app_db)
    assert len(await own.existing_chunks(a.tenant, did, hashes)) == 2


# --- jobs -------------------------------------------------------------------------------


async def test_jobs_lifecycle_and_retry(app_db: Database, a: World) -> None:
    jobs = PgJobs(app_db)
    ingest = await jobs.get(a.tenant, a.ingest_job)
    assert (ingest.kind, ingest.status, ingest.attempts) == ("ingest", "queued", 0)
    assert ingest.args["document_id"] == str(a.document_id)
    active = await jobs.active_for_collection(a.tenant, a.collection.id)
    assert a.ingest_job in {j.id for j in active}
    claimed = {c.id for c in await jobs.claim("w-life", 1000, timedelta(minutes=1))}
    assert a.ingest_job in claimed
    assert await jobs.fail(a.tenant, a.ingest_job, "w-life", "boom", timedelta(minutes=5))
    retry = await jobs.get(a.tenant, a.ingest_job)
    assert (retry.status, retry.last_error, retry.finished_at) == ("queued", "boom", None)
    assert retry.run_at > ingest.run_at
    assert not await jobs.fail(a.tenant, a.ingest_job, "w-life", "not owner any more", None)


async def test_job_fails_for_good(app_db: Database, a: World) -> None:
    jobs = PgJobs(app_db)
    job_id = await jobs.enqueue(
        a.tenant, NewJob("webhook", {"collection_id": str(a.collection.id)})
    )
    assert job_id in {c.id for c in await jobs.claim("w-fail", 1000, timedelta(minutes=1))}
    assert await jobs.fail(a.tenant, job_id, "w-fail", "final", None)
    record = await jobs.get(a.tenant, job_id)
    assert (record.status, record.last_error) == ("failed", "final")
    assert record.finished_at is not None


async def test_jobs_cross_tenant(b_db: Database, a: World, b: World) -> None:
    jobs = PgJobs(b_db)
    with pytest.raises(NotFoundError):
        await jobs.get(b.tenant, a.ingest_job)
    assert await jobs.active_for_collection(b.tenant, a.collection.id) == []
    lease = timedelta(minutes=1)
    assert not await jobs.extend_lease(b.tenant, a.ingest_job, "w", lease)
    assert not await jobs.set_progress(b.tenant, a.ingest_job, "w", 1, 1)
    assert not await jobs.complete(b.tenant, a.ingest_job, "w")
    assert not await jobs.fail(b.tenant, a.ingest_job, "w", "x", None)
    own = await jobs.enqueue(b.tenant, NewJob("gc"))
    assert (await jobs.get(b.tenant, own)).tenant_id == b.tenant
