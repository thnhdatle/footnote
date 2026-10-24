# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from footnote.adapters.postgres.chunks import PgChunks
from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.export import PgTenantExport
from footnote.adapters.postgres.index import PgIndex
from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database, Pool, fetch_one
from footnote.adapters.postgres.tenants import PgKeyResolver, PgTenantAdmin
from footnote.domain.documents import ChunkId, CollectionId, JobId
from footnote.domain.errors import NotFoundError, UnauthenticatedError
from footnote.domain.jobs import NewJob
from footnote.domain.settings import RetrievalSettings
from footnote.domain.tenancy import TenantId
from tests.integration.world import World, index_version, seed, upload_of

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
def db(app_pool: Pool, request: pytest.FixtureRequest) -> Database:
    return Database(app_pool, app_tenant_predicates=bool(request.param))


async def test_chunk_read_returns_current_memberships(a: World, db: Database) -> None:
    detail = await PgChunks(db).chunk(a.tenant, a.collection.id, a.chunk_ids[0])
    assert detail.id == a.chunk_ids[0]
    assert detail.document_id == a.document_id
    assert detail.text == "apple pie recipe"
    assert [(m.version, m.ordinal, m.page_from) for m in detail.memberships] == [(1, 0, 1)]


async def test_chunk_read_cross_tenant_is_chunk_not_found(a: World, b: World, db: Database) -> None:
    chunks = PgChunks(db)
    for tenant, collection, chunk in (
        (b.tenant, a.collection.id, a.chunk_ids[0]),
        (b.tenant, b.collection.id, a.chunk_ids[0]),
        (a.tenant, a.collection.id, b.chunk_ids[0]),
    ):
        with pytest.raises(NotFoundError) as err:
            await chunks.chunk(tenant, collection, chunk)
        assert err.value.code == "chunk_not_found"


async def test_chunk_read_superseded_chunk_is_not_found(a: World, app_db: Database) -> None:
    documents = PgDocuments(app_db)
    external_id = (await documents.get(a.tenant, a.collection.id, a.document_id)).external_id
    texts = ["banana bread recipe", "new cherry text"]
    upload = upload_of(a.collection, "\n\n".join(texts).encode(), external_id=external_id)
    _, _, _, chunk_ids = await index_version(app_db, a.collection, a.space, upload, texts)
    chunks = PgChunks(app_db)
    with pytest.raises(NotFoundError) as err:
        await chunks.chunk(a.tenant, a.collection.id, a.chunk_ids[0])
    assert err.value.code == "chunk_not_found"
    banana = await chunks.chunk(a.tenant, a.collection.id, a.chunk_ids[1])
    assert banana.id == chunk_ids[0]
    assert [(m.version, m.ordinal) for m in banana.memberships] == [(2, 0)]
    await PgCollections(app_db).mark_deleting(a.tenant, a.collection.id)
    with pytest.raises(NotFoundError):
        await chunks.chunk(a.tenant, a.collection.id, a.chunk_ids[1])


async def test_chunk_read_wrong_collection_is_not_found(a: World, app_db: Database) -> None:
    with pytest.raises(NotFoundError):
        await PgChunks(app_db).chunk(a.tenant, CollectionId(uuid.uuid4()), a.chunk_ids[0])
    with pytest.raises(NotFoundError):
        await PgChunks(app_db).chunk(a.tenant, a.collection.id, ChunkId(uuid.uuid4()))


async def test_find_by_external_id(a: World, b: World, db: Database) -> None:
    documents = PgDocuments(db)
    own = await documents.get(a.tenant, a.collection.id, a.document_id)
    found = await documents.find(a.tenant, a.collection.id, own.external_id)
    assert found is not None
    assert found.id == a.document_id
    assert await documents.find(a.tenant, a.collection.id, "missing") is None
    assert await documents.find(b.tenant, a.collection.id, own.external_id) is None
    assert await documents.find(b.tenant, b.collection.id, own.external_id) is None


async def test_update_enqueues_jobs_in_the_same_transaction(a: World, app_db: Database) -> None:
    collections = PgCollections(app_db)
    settings = a.collection.settings
    job = NewJob("reindex", {"collection_id": str(a.collection.id)})
    record, ids = await collections.update(a.tenant, a.collection.id, settings, jobs=(job,))
    assert record.id == a.collection.id
    assert len(ids) == 1
    stored = await PgJobs(app_db).get(a.tenant, ids[0])
    assert (stored.kind, stored.status, stored.args) == ("reindex", "queued", dict(job.args))


async def test_update_cross_tenant_writes_nothing(a: World, b: World, db: Database) -> None:
    changed = a.collection.settings.__class__(retrieval=RetrievalSettings(k=3))
    job = NewJob("reindex", {"collection_id": str(a.collection.id)})
    with pytest.raises(NotFoundError):
        await PgCollections(db).update(b.tenant, a.collection.id, changed, jobs=(job,))
    assert await _job_count(db, b.tenant, "reindex") == 0
    record = await PgCollections(db).get(a.tenant, a.collection.id)
    assert record.settings.retrieval.k == a.collection.settings.retrieval.k


async def test_mark_deleting_enqueues_and_cross_tenant_is_atomic(
    a: World, b: World, db: Database
) -> None:
    collections = PgCollections(db)
    job = NewJob("collection_delete", {"collection_id": str(a.collection.id)})
    with pytest.raises(NotFoundError):
        await collections.mark_deleting(b.tenant, a.collection.id, (job,))
    assert await _job_count(db, b.tenant, "collection_delete") == 0
    ids = await collections.mark_deleting(a.tenant, a.collection.id, (job,))
    assert len(ids) == 1
    assert (await PgJobs(db).get(a.tenant, JobId(ids[0]))).kind == "collection_delete"
    with pytest.raises(NotFoundError):
        await collections.get(a.tenant, a.collection.id)


async def test_tenant_mark_deleting_revokes_keys_and_enqueues(
    a: World, admin_db: Database, app_db: Database
) -> None:
    job = NewJob("tenant_delete", {"tenant_id": str(a.tenant)})
    ids = await PgTenantAdmin(admin_db).mark_deleting(a.tenant, (job,))
    assert len(ids) == 1
    assert (await PgJobs(app_db).get(a.tenant, ids[0])).kind == "tenant_delete"
    with pytest.raises(UnauthenticatedError):
        await PgKeyResolver(app_db).resolve(a.key)
    with pytest.raises(NotFoundError):
        await PgTenantAdmin(admin_db).mark_deleting(TenantId(uuid.uuid4()), (job,))


async def _job_count(db: Database, tenant: TenantId, kind: str) -> int:
    async with db.tenant_tx(tenant) as conn:
        row = await fetch_one(
            conn,
            "SELECT count(*) AS n FROM footnote.jobs WHERE kind = %(k)s",
            {"k": kind},
        )
    assert row is not None
    return int(row["n"])


async def _export(admin_db: Database, tenant: TenantId, **flags: bool) -> list[tuple[str, Any]]:
    exporter = PgTenantExport(admin_db)
    options = {"include_vectors": False, "include_originals": False, **flags}
    return [(r.type, r.data) async for r in exporter.export(tenant, **options)]


async def test_export_streams_the_tenant_in_dependency_order(
    a: World, b: World, admin_db: Database
) -> None:
    records = await _export(admin_db, a.tenant)
    kinds = [k for k, _ in records]
    assert kinds[0] == "tenant"
    order = ["tenant", "collection", "document", "version", "elements", "chunk", "membership"]
    assert sorted(set(kinds), key=order.index) == order
    assert kinds == sorted(kinds, key=order.index)
    collection = next(d for k, d in records if k == "collection")
    assert collection["id"] == a.collection.id
    assert "webhook_secret" not in collection
    assert collection["webhook_url"] == "https://example.com/hook"
    chunks = {d["id"] for k, d in records if k == "chunk"}
    assert chunks == set(a.chunk_ids)
    elements = next(d for k, d in records if k == "elements")
    assert [e["text"] for e in elements["elements"]] == a.texts
    version = next(d for k, d in records if k == "version")
    assert "original_base64" not in version
    flat = repr(records)
    assert str(b.tenant) not in flat
    assert str(b.collection.id) not in flat


async def test_export_optional_vectors_and_originals(a: World, admin_db: Database) -> None:
    records = await _export(admin_db, a.tenant, include_vectors=True, include_originals=True)
    vectors = [d for k, d in records if k == "vector"]
    assert {v["chunk_id"] for v in vectors} == set(a.chunk_ids)
    assert all(len(v["embedding"]) == a.space.dimensions for v in vectors)
    assert {v["space"] for v in vectors} == {a.space.name}
    version = next(d for k, d in records if k == "version")
    original = base64.b64decode(version["original_base64"])
    assert original == "\n\n".join(a.texts).encode()


async def test_export_unknown_tenant_is_not_found(admin_db: Database) -> None:
    with pytest.raises(NotFoundError):
        await _export(admin_db, TenantId(uuid.uuid4()))


async def test_export_survives_an_early_close(a: World, admin_db: Database) -> None:
    stream = PgTenantExport(admin_db).export(
        a.tenant, include_vectors=False, include_originals=False
    )
    first = await anext(stream)
    assert first.type == "tenant"
    await stream.aclose()  # type: ignore[attr-defined]
    assert next(k for k, _ in await _export(admin_db, a.tenant)) == "tenant"


async def test_index_unchanged_by_reads(a: World, app_db: Database) -> None:
    count = await PgIndex(app_db).current_chunk_count(a.tenant, a.document_id)
    assert count == len(a.chunk_ids)
