# SPDX-License-Identifier: Apache-2.0
"""PgSearchStore: lexical modes, vector legs, filters, display data, snapshot isolation."""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest
import pytest_asyncio

from footnote.adapters.postgres.collection_repo import PgCollections
from footnote.adapters.postgres.documents import PgDocuments
from footnote.adapters.postgres.search import PgSearchStore, SnapshotSettings
from footnote.adapters.postgres.session import Database, Pool
from footnote.app.search import SearchConfig, SearchRequest, SearchService
from footnote.domain.documents import DocumentId
from footnote.domain.embedding import l2_normalise
from footnote.domain.errors import NotFoundError
from footnote.domain.ports import LexicalQuery, SearchFilters, VectorQuery
from footnote.domain.settings import LexicalMatch, LexicalRank, SearchMode
from tests.integration.world import World, hash_vector, index_version, seed, upload_of
from tests.unit.app.fakes import FakeClock, FakeEmbedderFactory

pytestmark = pytest.mark.integration

TEXTS = ["the quick brown fox", "a lazy dog sleeps", "quick dog runs fast"]


def _q(text: str, match: LexicalMatch, rank: LexicalRank = LexicalRank.TS_RANK) -> LexicalQuery:
    return LexicalQuery(text, match, rank, 50)


@pytest_asyncio.fixture
async def w(admin_db: Database, app_db: Database) -> AsyncIterator[World]:
    """A tenant whose collection holds TEXTS (one document, pages 1..3)."""
    yield await seed(admin_db, app_db, f"search-{uuid.uuid4().hex[:8]}", TEXTS)


@pytest.fixture(params=[True, False], ids=["predicates", "rls-only"])
def store(request: pytest.FixtureRequest, app_pool: Pool) -> PgSearchStore:
    """Store with and without the app-side tenant predicates."""
    return PgSearchStore(Database(app_pool, app_tenant_predicates=bool(request.param)))


async def test_collection_view(store: PgSearchStore, w: World) -> None:
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        view = await snap.collection()
        assert view is await snap.collection()  # read once per snapshot
    assert (view.id, view.chunk_count, view.active_space) == (w.collection.id, 3, w.space)
    assert view.settings == w.collection.settings


async def test_lexical_modes_when_no_chunk_has_every_word(store: PgSearchStore, w: World) -> None:
    ids = dict(zip(TEXTS, w.chunk_ids, strict=True))
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        any_hits = await snap.lexical(_q("quick dog jumps", LexicalMatch.ANY), SearchFilters())
        cd_hits = await snap.lexical(
            _q("quick dog jumps", LexicalMatch.ANY, LexicalRank.TS_RANK_CD), SearchFilters()
        )
        all_hits = await snap.lexical(_q("quick dog jumps", LexicalMatch.ALL), SearchFilters())
        web_and = await snap.lexical(_q("quick dog jumps", LexicalMatch.WEBSEARCH), SearchFilters())
        web_or = await snap.lexical(_q("quick or jumps", LexicalMatch.WEBSEARCH), SearchFilters())
        nothing = await snap.lexical(_q("!!! ???", LexicalMatch.ANY), SearchFilters())
        limited = await snap.lexical(
            LexicalQuery("quick dog", LexicalMatch.ANY, LexicalRank.TS_RANK, 1), SearchFilters()
        )
    assert {h.chunk_id for h in any_hits} == set(w.chunk_ids)
    assert any_hits[0].chunk_id == ids["quick dog runs fast"]
    assert any_hits[0].score > any_hits[1].score > 0
    assert cd_hits[0].chunk_id == ids["quick dog runs fast"]
    assert all_hits == []
    assert web_and == []
    assert {h.chunk_id for h in web_or} == {ids["the quick brown fox"], ids["quick dog runs fast"]}
    assert nothing == []
    assert [h.chunk_id for h in limited] == [ids["quick dog runs fast"]]


async def test_vector_exact_and_hnsw_paths(store: PgSearchStore, w: World) -> None:
    query = l2_normalise(hash_vector(TEXTS[1]))
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        exact = await snap.vector(VectorQuery(query, 10, exact=True), SearchFilters())
        hnsw = await snap.vector(VectorQuery(query, 10, exact=False), SearchFilters())
        sims = await snap.similarities(w.chunk_ids, query)
    assert exact[0].chunk_id == w.chunk_ids[1]
    assert exact[0].score == pytest.approx(1.0, abs=1e-6)
    assert [h.chunk_id for h in hnsw] == [h.chunk_id for h in exact]
    assert all(a.score >= b.score for a, b in itertools.pairwise(exact))
    assert set(sims) == set(w.chunk_ids)
    assert sims[w.chunk_ids[1]] == pytest.approx(1.0, abs=1e-6)
    assert sims[w.chunk_ids[0]] == pytest.approx(
        exact[[h.chunk_id for h in exact].index(w.chunk_ids[0])].score
    )


async def test_hits_display_data_in_input_order(
    store: PgSearchStore, app_db: Database, w: World
) -> None:
    await PgDocuments(app_db).update(w.tenant, w.collection.id, w.document_id, title="Shown")
    order = [w.chunk_ids[2], w.chunk_ids[0]]
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        hits = await snap.hits([*order, uuid.uuid4()])  # type: ignore[list-item]
        assert await snap.hits([]) == []
        assert await snap.similarities([], [1.0] * 8) == {}
    assert [h.chunk_id for h in hits] == order
    first = hits[0]
    assert (first.text, first.title, first.heading_path) == (TEXTS[2], "Shown", ("Intro",))
    assert (first.page_from, first.page_to) == (3, 3)
    assert first.embedded_text == "Doc\nIntro\n\n" + TEXTS[2]
    assert first.document_id == w.document_id


async def test_filters(store: PgSearchStore, app_db: Database, w: World) -> None:
    other_doc, _, _, other_chunks = await index_version(
        app_db,
        w.collection,
        w.space,
        upload_of(w.collection, b"second", metadata={"lang": "de", "team": "x"}),
        ["quick second document"],
    )
    q = _q("quick", LexicalMatch.ANY)
    vec = VectorQuery(l2_normalise(hash_vector("quick")), 50, exact=True)

    async def ids(filters: SearchFilters) -> tuple[set[object], set[object]]:
        async with store.snapshot(w.tenant, w.collection.id) as snap:
            lex = {h.chunk_id for h in await snap.lexical(q, filters)}
            vector = {h.chunk_id for h in await snap.vector(vec, filters)}
        return lex, vector

    lex, vector = await ids(SearchFilters(metadata={"lang": "de"}))
    assert lex == vector - set(w.chunk_ids) == set(other_chunks)
    lex, _ = await ids(SearchFilters(metadata={"lang": {"in": ["de", "en"]}}))
    assert lex == {w.chunk_ids[0], w.chunk_ids[2], other_chunks[0]}
    lex, vector = await ids(SearchFilters(metadata={"lang": "en", "team": "x"}))
    assert lex == vector == set()
    lex, vector = await ids(SearchFilters(document_ids=(DocumentId(other_doc),)))
    assert vector == set(other_chunks)
    lex, _ = await ids(SearchFilters(created_after="2000-01-01T00:00:00Z"))
    assert other_chunks[0] in lex
    lex, vector = await ids(SearchFilters(created_before="2000-01-01T00:00:00Z"))
    assert lex == vector == set()


async def test_only_current_versions_and_snapshot_isolation(
    store: PgSearchStore, app_db: Database, w: World
) -> None:
    ext = (await PgDocuments(app_db).get(w.tenant, w.collection.id, w.document_id)).external_id
    sleeping = _q("sleeps", LexicalMatch.ANY)
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        assert len(await snap.lexical(sleeping, SearchFilters())) == 1
        # A version swap commits while the snapshot is open: the snapshot keeps the old view.
        await index_version(
            app_db,
            w.collection,
            w.space,
            upload_of(w.collection, b"v2", external_id=ext),
            [TEXTS[0], "an awake cat"],
        )
        assert len(await snap.lexical(sleeping, SearchFilters())) == 1
        assert (await snap.collection()).chunk_count == 3
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        assert await snap.lexical(sleeping, SearchFilters()) == []  # old chunk not yet GC'd
        vec = VectorQuery(l2_normalise(hash_vector(TEXTS[1])), 50, exact=True)
        assert w.chunk_ids[1] not in {h.chunk_id for h in await snap.vector(vec, SearchFilters())}
        assert (await snap.collection()).chunk_count == 2


async def test_snapshot_is_read_only(store: PgSearchStore, w: World) -> None:
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        conn = snap._conn
        cur = await conn.execute(
            "SELECT current_setting('transaction_isolation'), "
            "current_setting('transaction_read_only'), current_setting('plan_cache_mode'), "
            "current_setting('hnsw.iterative_scan')"
        )
        assert await cur.fetchone() == (
            "repeatable read",
            "on",
            "force_custom_plan",
            "relaxed_order",
        )


async def test_cross_tenant_and_deleting_collections_are_absent(
    store: PgSearchStore, admin_db: Database, app_db: Database, w: World
) -> None:
    intruder = await seed(admin_db, app_db, f"intruder-{uuid.uuid4().hex[:8]}", ["quick"])
    async with store.snapshot(intruder.tenant, w.collection.id) as snap:
        with pytest.raises(NotFoundError):
            await snap.collection()
        with pytest.raises(NotFoundError):
            await snap.lexical(_q("quick", LexicalMatch.ANY), SearchFilters())
        assert await snap.hits(w.chunk_ids) == []
    await PgCollections(app_db).mark_deleting(w.tenant, w.collection.id)
    async with store.snapshot(w.tenant, w.collection.id) as snap:
        with pytest.raises(NotFoundError):
            await snap.collection()


async def test_search_service_end_to_end(app_pool: Pool, w: World) -> None:
    service = SearchService(
        PgSearchStore(Database(app_pool), SnapshotSettings(ef_search=40)),
        FakeEmbedderFactory(),
        None,
        FakeClock(),
        SearchConfig(exact_scan_max=0),  # force the HNSW path
    )
    result = await service.search(
        w.tenant,
        w.collection.id,
        SearchRequest(TEXTS[2], mode=SearchMode.HYBRID, k=2, need_similarity=True),
    )
    assert result.hits[0].chunk_id == w.chunk_ids[2]
    assert result.hits[0].ranks == {"lexical": 1, "vector": 1}
    assert result.similarities[w.chunk_ids[2]] == pytest.approx(1.0, abs=1e-6)
    assert result.exact_scan_reason in (None, "fallback")
    lexical = await service.search(
        w.tenant, w.collection.id, SearchRequest("lazy", mode=SearchMode.LEXICAL)
    )
    assert [h.chunk_id for h in lexical.hits] == [w.chunk_ids[1]]
    unchanged = replace(result.collection.settings)
    assert unchanged == w.collection.settings
