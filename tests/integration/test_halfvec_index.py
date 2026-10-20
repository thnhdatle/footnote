# SPDX-License-Identifier: Apache-2.0
"""Spaces above 2000 dimensions use a partial `halfvec` expression HNSW index."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

import psycopg
import pytest

from footnote.adapters.postgres.search import PgSearchStore
from footnote.adapters.postgres.session import Database, Pool
from footnote.domain.documents import CollectionId
from footnote.domain.embedding import l2_normalise
from footnote.domain.ports import SearchFilters, VectorQuery
from footnote.domain.tenancy import TenantId
from tests.integration.conftest import PgInfo

pytestmark = pytest.mark.integration

HALF_DIMS = 2001
ROWS = 6000


def _register(conn: psycopg.Connection[tuple[object, ...]], name: str, dims: int) -> int:
    row = conn.execute(
        "SELECT footnote.footnote_register_space(%s, 'fake', 'model', %s, false, true, 0.4)",
        (name, dims),
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


@pytest.mark.parametrize(
    ("dims", "index_type"), [(2000, "vector"), (2001, "halfvec"), (4000, "halfvec")]
)
def test_index_type_switches_above_2000_dims(pg: PgInfo, dims: int, index_type: str) -> None:
    name = f"it-dims-{dims}-{uuid.uuid4().hex[:6]}"
    with psycopg.connect(pg.admin, autocommit=True) as conn:
        space_id = _register(conn, name, dims)
        stored = conn.execute(
            "SELECT index_type FROM footnote.embedding_spaces WHERE id = %s", (space_id,)
        ).fetchone()
        index = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
            (f"chunk_vectors_hnsw_s{space_id}",),
        ).fetchone()
    assert stored is not None
    assert stored[0] == index_type
    assert index is not None
    assert f"((embedding)::{index_type}({dims})) {index_type}_cosine_ops" in str(index[0])
    assert f"WHERE (space_id = {space_id})" in str(index[0])


@dataclass(frozen=True)
class Loaded:
    """A collection of ROWS random HALF_DIMS-dimension vectors."""

    tenant: uuid.UUID
    collection: uuid.UUID
    space_id: int


def _load(pg: PgInfo) -> Loaded:
    tenant, collection, document, version = (uuid.uuid4() for _ in range(4))
    with psycopg.connect(pg.admin, autocommit=True) as conn:
        space_id = _register(conn, f"it-half-{uuid.uuid4().hex[:6]}", HALF_DIMS)
        p = {
            "t": tenant,
            "c": collection,
            "d": document,
            "v": version,
            "s": space_id,
            "n": ROWS,
            "dims": HALF_DIMS,
        }
        with conn.transaction():
            conn.execute("SELECT setseed(0.42)")
            conn.execute("INSERT INTO footnote.tenants (id, name) VALUES (%(t)s, 'half')", p)
            conn.execute(
                "INSERT INTO footnote.collections (tenant_id, id, name, active_space_id, "
                "chunk_count) VALUES (%(t)s, %(c)s, 'half', %(s)s, %(n)s)",
                p,
            )
            conn.execute(
                "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id, "
                "status) VALUES (%(t)s, %(d)s, %(c)s, 'half', 'ready')",
                p,
            )
            conn.execute(
                "INSERT INTO footnote.document_versions (tenant_id, id, document_id, version, "
                "source_sha256, media_type, bytes, chunking_hash, status) VALUES (%(t)s, %(v)s, "
                "%(d)s, 1, sha256('half'::bytea), 'text/plain', 0, 'x', 'ready')",
                p,
            )
            conn.execute(
                "UPDATE footnote.documents SET current_version_id = %(v)s WHERE id = %(d)s", p
            )
            conn.execute(
                "INSERT INTO footnote.chunks (tenant_id, id, collection_id, document_id, "
                "content_hash, text, embedded_text, token_count, fts_config) "
                "SELECT %(t)s, gen_random_uuid(), %(c)s, %(d)s, "
                "encode(sha256(convert_to(i::text, 'UTF8')), 'hex'), 'half ' || i, "
                "'half ' || i, 2, 'simple' FROM generate_series(1, %(n)s) AS i",
                p,
            )
            conn.execute(
                "INSERT INTO footnote.version_chunks (tenant_id, version_id, ordinal, chunk_id) "
                "SELECT tenant_id, %(v)s, row_number() OVER (ORDER BY id) - 1, id "
                "FROM footnote.chunks WHERE tenant_id = %(t)s",
                p,
            )
            conn.execute(
                "INSERT INTO footnote.chunk_vectors (tenant_id, chunk_id, space_id, "
                "collection_id, embedding) SELECT tenant_id, id, %(s)s, collection_id, "
                "ARRAY(SELECT random() - 0.5 FROM generate_series(1, %(dims)s) "
                "WHERE c.id IS NOT NULL)::real[]::vector "
                "FROM footnote.chunks c WHERE tenant_id = %(t)s",
                p,
            )
        conn.execute(
            "ANALYZE footnote.chunks, footnote.chunk_vectors, footnote.documents, "
            "footnote.version_chunks, footnote.collections"
        )
    return Loaded(tenant, collection, space_id)


async def test_explain_of_halfvec_space_uses_partial_halfvec_hnsw(
    pg: PgInfo, app_pool: Pool
) -> None:
    loaded = _load(pg)
    store = PgSearchStore(Database(app_pool))
    rng = random.Random(7)  # noqa: S311 - test data
    probe = [rng.uniform(-1, 1) for _ in range(HALF_DIMS)]
    query = VectorQuery(l2_normalise(probe), 50, exact=False)
    async with store.snapshot(TenantId(loaded.tenant), CollectionId(loaded.collection)) as snap:
        view = await snap.collection()
        plan = await snap.explain_vector(query, SearchFilters())
        hits = await snap.vector(query, SearchFilters())
    assert view.active_space.index_type == "halfvec"
    assert view.active_space.dimensions == HALF_DIMS
    assert f"chunk_vectors_hnsw_s{loaded.space_id}" in plan, plan
    assert "Seq Scan on chunk_vectors" not in plan, plan
    assert len(hits) == 50
