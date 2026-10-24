# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace

import psycopg
import pytest
import pytest_asyncio

from footnote.adapters.postgres.search import PgSearchStore
from footnote.adapters.postgres.session import Database, Pool
from footnote.app.search import SearchConfig, SearchRequest, SearchService
from footnote.domain.documents import CollectionId
from footnote.domain.embedding import l2_normalise
from footnote.domain.ports import SearchFilters, VectorQuery
from footnote.domain.settings import SearchMode
from footnote.domain.tenancy import TenantId
from tests.integration.conftest import PgInfo, drop_extra_data
from tests.integration.world import DIMS, World, hash_vector, seed
from tests.unit.app.fakes import FakeClock, FakeEmbedderFactory

pytestmark = pytest.mark.integration

FILLER_VECTORS = 100_000
SCALE_SPACE = "it-scale-8"
FIVE = ["solar wind", "lunar tide", "stellar dust", "comet tail", "nebula glow"]


@dataclass(frozen=True)
class Filler:
    tenant: uuid.UUID
    collection: uuid.UUID
    space_id: int
    seconds: float


def _register(conn: psycopg.Connection[tuple[object, ...]]) -> int:
    row = conn.execute(
        "SELECT footnote.footnote_register_space(%s, 'fake', 'scale-model', %s, false, true, 0.4)",
        (SCALE_SPACE, DIMS),
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


@pytest.fixture(scope="module")
def filler(pg: PgInfo) -> Iterator[Filler]:
    tenant, collection, document, version = (uuid.uuid4() for _ in range(4))
    started = time.monotonic()
    with psycopg.connect(pg.admin, autocommit=True) as conn:
        space_id = _register(conn)
        conn.execute(f"DROP INDEX footnote.chunk_vectors_hnsw_s{space_id}")
        with conn.transaction():
            _bulk_load(
                conn,
                {
                    "t": tenant,
                    "c": collection,
                    "d": document,
                    "v": version,
                    "s": space_id,
                    "n": FILLER_VECTORS,
                },
            )
        conn.execute("SET maintenance_work_mem = '128MB'")
        conn.execute("SET max_parallel_maintenance_workers = 0")
        assert _register(conn) == space_id
        conn.execute(
            "ANALYZE footnote.chunks, footnote.chunk_vectors, footnote.documents, "
            "footnote.version_chunks, footnote.collections"
        )
    yield Filler(tenant, collection, space_id, time.monotonic() - started)
    drop_extra_data(pg)


def _bulk_load(conn: psycopg.Connection[tuple[object, ...]], p: dict[str, object]) -> None:
    conn.execute("SELECT setseed(0.42)")
    conn.execute("INSERT INTO footnote.tenants (id, name) VALUES (%(t)s, 'filler')", p)
    conn.execute(
        "INSERT INTO footnote.collections (tenant_id, id, name, active_space_id, chunk_count) "
        "VALUES (%(t)s, %(c)s, 'filler', %(s)s, %(n)s)",
        p,
    )
    conn.execute(
        "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id, status) "
        "VALUES (%(t)s, %(d)s, %(c)s, 'filler', 'ready')",
        p,
    )
    conn.execute(
        "INSERT INTO footnote.document_versions (tenant_id, id, document_id, version, "
        "source_sha256, media_type, bytes, chunking_hash, status) VALUES (%(t)s, %(v)s, %(d)s, "
        "1, sha256('filler'::bytea), 'text/plain', 0, 'x', 'ready')",
        p,
    )
    conn.execute("UPDATE footnote.documents SET current_version_id = %(v)s WHERE id = %(d)s", p)
    conn.execute(
        "INSERT INTO footnote.chunks (tenant_id, id, collection_id, document_id, "
        "content_hash, text, embedded_text, token_count, fts_config) "
        "SELECT %(t)s, gen_random_uuid(), %(c)s, %(d)s, "
        "encode(sha256(convert_to(i::text, 'UTF8')), 'hex'), 'filler ' || i, "
        "'filler ' || i, 2, 'simple' FROM generate_series(1, %(n)s) AS i",
        p,
    )
    conn.execute(
        "INSERT INTO footnote.version_chunks (tenant_id, version_id, ordinal, chunk_id) "
        "SELECT tenant_id, %(v)s, row_number() OVER (ORDER BY id) - 1, id "
        "FROM footnote.chunks WHERE tenant_id = %(t)s",
        p,
    )
    conn.execute(
        "INSERT INTO footnote.chunk_vectors (tenant_id, chunk_id, space_id, collection_id, "
        "embedding) SELECT tenant_id, id, %(s)s, collection_id, "
        "ARRAY[random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5, "
        "random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5]::real[]::vector "
        "FROM footnote.chunks WHERE tenant_id = %(t)s",
        p,
    )


@pytest_asyncio.fixture
async def five(admin_db: Database, app_db: Database, filler: Filler) -> AsyncIterator[World]:
    yield await seed(admin_db, app_db, f"five-{uuid.uuid4().hex[:8]}", FIVE, SCALE_SPACE)


def _service(app_pool: Pool) -> SearchService:
    return SearchService(
        PgSearchStore(Database(app_pool)),
        FakeEmbedderFactory(),
        None,
        FakeClock(),
        SearchConfig(exact_scan_max=0),
    )


async def test_five_chunk_collection_returns_all_five(app_pool: Pool, five: World) -> None:
    result = await _service(app_pool).search(
        five.tenant, five.collection.id, SearchRequest("solar wind", mode=SearchMode.VECTOR, k=5)
    )
    assert {h.chunk_id for h in result.hits} == set(five.chunk_ids)
    assert result.hits[0].chunk_id == five.chunk_ids[0]
    assert result.exact_scan_reason in (None, "fallback")


async def test_selective_filter_returns_k_vector_rows_or_falls_back(
    app_pool: Pool, five: World
) -> None:
    filters = SearchFilters(metadata={"lang": "en"})
    result = await _service(app_pool).search(
        five.tenant,
        five.collection.id,
        SearchRequest("comet tail", mode=SearchMode.VECTOR, k=5, filters=filters),
    )
    k_vector = result.collection.settings.retrieval.k_vector
    assert len(result.ranking) == 5 < k_vector
    assert result.exact_scan_reason == "fallback"
    assert result.hits[0].chunk_id == five.chunk_ids[3]


async def test_explain_uses_partial_hnsw_and_btree_never_seq_scan(
    app_pool: Pool, five: World, filler: Filler
) -> None:
    store = PgSearchStore(Database(app_pool))
    query = VectorQuery(l2_normalise(hash_vector("stellar dust")), 50, exact=False)
    async with store.snapshot(TenantId(filler.tenant), CollectionId(filler.collection)) as snap:
        hnsw = await snap.explain_vector(query, SearchFilters())
        large = await snap.vector(query, SearchFilters())
    assert f"chunk_vectors_hnsw_s{filler.space_id}" in hnsw, hnsw
    assert len(large) == 50
    async with store.snapshot(five.tenant, five.collection.id) as snap:
        exact = await snap.explain_vector(replace(query, exact=True), SearchFilters())
    assert "chunk_vectors_tenant_collection_space_idx" in exact, exact
    for plan in (hnsw, exact):
        assert "Seq Scan on chunk_vectors" not in plan, plan
    assert filler.seconds < 300
