# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the search use case, driven through in-memory fakes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import pytest

from footnote.app.search import SearchConfig, SearchOverrides, SearchRequest, SearchService
from footnote.domain.documents import ChunkId, CollectionId
from footnote.domain.embedding import l2_normalise
from footnote.domain.errors import ValidationError
from footnote.domain.ports import LexicalQuery, SearchFilters, VectorQuery
from footnote.domain.retrieval import FusionConfig, RankedList, rrf
from footnote.domain.settings import (
    CollectionSettings,
    LexicalMatch,
    LexicalRank,
    RetrievalSettings,
    SearchMode,
)
from footnote.domain.tenancy import TenantId

from .fakes import (
    FakeClock,
    FakeCollection,
    FakeEmbedderFactory,
    FakeReranker,
    FakeSearchStore,
    hash_vector,
    make_space,
)

TENANT = TenantId(UUID(int=7))
COLLECTION = CollectionId(UUID(int=99))
TEXTS = ["alpha beta gamma", "beta delta", "gamma epsilon zeta", "unrelated words here"]
C1, C2, C3, C4 = (ChunkId(UUID(int=n)) for n in (1, 2, 3, 4))


@dataclass
class Rig:
    service: SearchService
    store: FakeSearchStore
    embedders: FakeEmbedderFactory
    reranker: FakeReranker | None
    collection: FakeCollection
    events: list[str]

    async def search(self, query: str, **kw: Any) -> Any:
        return await self.service.search(TENANT, COLLECTION, SearchRequest(query, **kw))


def build(
    *,
    rerank: bool = True,
    with_reranker: bool = True,
    config: SearchConfig | None = None,
    texts: list[str] = TEXTS,
    **retrieval: Any,
) -> Rig:
    events: list[str] = []
    settings = CollectionSettings(
        retrieval=replace(RetrievalSettings(), rerank=rerank, **retrieval)
    )
    collection = FakeCollection(make_space(), settings)
    for text in texts:
        collection.add(text)
    store = FakeSearchStore(collection, events)
    embedders = FakeEmbedderFactory()
    reranker = FakeReranker(events=events) if with_reranker else None
    service = SearchService(store, embedders, reranker, FakeClock(), config)
    return Rig(service, store, embedders, reranker, collection, events)


async def test_search_vector_mode_ranks_by_cosine_and_skips_lexical() -> None:
    rig = build(rerank=False)
    result = await rig.search("alpha beta gamma", mode=SearchMode.VECTOR)
    assert result.hits[0].chunk_id == C1
    assert result.hits[0].scores["vector"] == pytest.approx(1.0)
    assert "lexical" not in rig.events
    assert set(result.stage_seconds) == {"embed_query", "vector"}
    assert rig.embedders.embedder.query_calls == ["alpha beta gamma"]
    assert not result.reranked


async def test_search_lexical_mode_skips_embedding_and_vector() -> None:
    rig = build(rerank=False)
    result = await rig.search("beta", mode=SearchMode.LEXICAL)
    assert [h.chunk_id for h in result.hits] == [C1, C2]
    assert [h.ranks for h in result.hits] == [{"lexical": 1}, {"lexical": 2}]
    assert rig.embedders.embedder.query_calls == []
    assert "vector" not in rig.events
    assert set(result.stage_seconds) == {"lexical"}
    assert result.exact_scan_reason is None


async def test_search_hybrid_order_matches_domain_rrf() -> None:
    rig = build(rerank=False)
    query = "beta gamma"
    result = await rig.search(query, mode=SearchMode.HYBRID)
    retrieval = RetrievalSettings()
    async with rig.store.snapshot(TENANT, COLLECTION) as snap:
        lexical = await snap.lexical(
            LexicalQuery(query, retrieval.lexical_match, retrieval.lexical_rank, 50),
            SearchFilters(),
        )
        vector = await snap.vector(
            VectorQuery(l2_normalise(hash_vector(query)), 50, True), SearchFilters()
        )
    fused = rrf(
        [RankedList("lexical", tuple(lexical)), RankedList("vector", tuple(vector))],
        FusionConfig(rrf_k=60, weights={"lexical": 0.2, "vector": 1.0}),
    )
    assert [h.chunk_id for h in result.hits] == [f.chunk_id for f in fused]
    assert [h.rrf_score for h in result.hits] == [f.rrf_score for f in fused]
    assert set(result.hits[0].ranks) <= {"lexical", "vector"}


async def test_search_default_mode_comes_from_config() -> None:
    rig = build(rerank=False, config=SearchConfig(default_mode=SearchMode.LEXICAL))
    await rig.search("beta")
    assert "vector" not in rig.events


async def test_search_rerank_orders_by_logit_and_records_scores() -> None:
    rig = build(rerank=True)
    result = await rig.search("beta gamma", mode=SearchMode.LEXICAL)
    assert rig.reranker is not None
    query, passages = rig.reranker.calls[0]
    assert query == "beta gamma"
    assert passages == ["alpha beta gamma", "beta delta", "gamma epsilon zeta"]
    assert result.reranked
    assert result.rerank_logits == [1.0, 0.0, 0.0]
    assert [h.rerank_score for h in result.hits] == [1.0, 0.0, 0.0]
    assert "rerank" in result.stage_seconds


async def test_search_rerank_reorders_head_stably() -> None:
    rig = build(rerank=True, k=2, rerank_n=2)
    # chunk 3 is the nearest vector, chunk 2 the second; the reranker prefers chunk 2.
    rig.collection.chunks[2].vector = l2_normalise(hash_vector("beta"))
    near = hash_vector("beta")
    near[0] += 0.1
    rig.collection.chunks[1].vector = l2_normalise(near)
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert [h.chunk_id for h in result.hits] == [C2, C3]
    assert result.rerank_logits[:2] == [-1.0, 0.0]


async def test_search_rerank_request_overrides_collection_setting() -> None:
    on = build(rerank=False)
    result = await on.search("beta", mode=SearchMode.LEXICAL, rerank=True)
    assert result.reranked
    off = build(rerank=True)
    result = await off.search("beta", mode=SearchMode.LEXICAL, rerank=False)
    assert not result.reranked
    assert off.reranker is not None
    assert off.reranker.calls == []


async def test_search_without_reranker_never_reranks() -> None:
    rig = build(rerank=True, with_reranker=False)
    result = await rig.search("beta", mode=SearchMode.LEXICAL, rerank=True)
    assert not result.reranked
    assert result.rerank_logits == []
    assert rig.service.reranker is None


async def test_search_rerank_with_no_candidates_skips_reranker() -> None:
    rig = build(rerank=True)
    result = await rig.search("nomatch", mode=SearchMode.LEXICAL, need_similarity=True)
    assert result.hits == []
    assert result.reranked  # rerank stage was active, so the rerank gate applies
    assert result.similarities == {}
    assert rig.reranker is not None
    assert rig.reranker.calls == []


@pytest.mark.parametrize("k", [0, -1, 101])
async def test_search_k_out_of_range_rejected(k: int) -> None:
    rig = build()
    with pytest.raises(ValidationError) as info:
        await rig.search("beta", k=k)
    assert info.value.field == "k"


@pytest.mark.parametrize("k", [1, 100])
async def test_search_k_bounds_accepted(k: int) -> None:
    rig = build(rerank=False)
    result = await rig.search("beta", k=k, mode=SearchMode.LEXICAL)
    assert len(result.hits) == min(k, 2)


@pytest.mark.parametrize("query", ["", "   \n\t"])
async def test_search_empty_query_rejected_before_store_is_opened(query: str) -> None:
    rig = build()
    with pytest.raises(ValidationError) as info:
        await rig.search(query)
    assert info.value.field == "query"
    assert rig.events == []


async def test_search_small_collection_uses_exact_scan() -> None:
    rig = build(rerank=False, config=SearchConfig(exact_scan_max=4))
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert result.exact_scan_reason == "small_collection"
    assert rig.store.vector_calls == [True]


async def test_search_large_collection_uses_hnsw_without_reason() -> None:
    rig = build(rerank=False, k_vector=4, config=SearchConfig(exact_scan_max=3))
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert result.exact_scan_reason is None
    assert rig.store.vector_calls == [False]
    assert len(result.hits) == 4


async def test_search_hnsw_short_result_falls_back_to_exact() -> None:
    rig = build(rerank=False, k_vector=4, config=SearchConfig(exact_scan_max=3))
    rig.store.hnsw_rows = 2
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert result.exact_scan_reason == "fallback"
    assert rig.store.vector_calls == [False, True]
    assert len(result.hits) == 4


async def test_search_need_similarity_embeds_query_in_lexical_mode_without_rerank() -> None:
    rig = build(rerank=False)
    result = await rig.search("beta", mode=SearchMode.LEXICAL, need_similarity=True)
    assert rig.embedders.embedder.query_calls == ["beta"]
    assert set(result.similarities) == {C1, C2}
    assert "vector" not in rig.events
    assert set(result.stage_seconds) == {"embed_query", "lexical"}


async def test_search_need_similarity_skips_embedding_in_lexical_mode_with_rerank() -> None:
    rig = build(rerank=True)
    result = await rig.search("beta", mode=SearchMode.LEXICAL, need_similarity=True)
    assert rig.embedders.embedder.query_calls == []
    assert result.similarities == {}


async def test_search_similarities_are_cosine_of_normalised_query() -> None:
    rig = build(rerank=False)
    result = await rig.search("alpha beta gamma", mode=SearchMode.VECTOR, need_similarity=True)
    assert result.similarities[C1] == pytest.approx(1.0)
    assert all(-1.0 - 1e-9 <= s <= 1.0 + 1e-9 for s in result.similarities.values())


async def test_search_depth_ranking_is_rerank_head_then_rrf_order() -> None:
    rig = build(rerank=True, rerank_n=2, k=1)
    rig.collection.chunks[2].vector = l2_normalise(hash_vector("beta"))
    near = hash_vector("beta")
    near[0] += 0.1
    rig.collection.chunks[1].vector = l2_normalise(near)
    plain = await build(rerank=False).search("beta", mode=SearchMode.VECTOR, depth=4)
    result = await rig.search("beta", mode=SearchMode.VECTOR, depth=4)
    assert result.ranking[:2] == [C2, C3]
    assert result.ranking[2:] == [c for c in plain.ranking if c not in (C2, C3)]
    assert len(result.ranking) == 4
    assert [h.chunk_id for h in result.hits] == [C2]
    assert rig.reranker is not None
    assert len(rig.reranker.calls[0][1]) == 2


async def test_search_ranking_without_depth_is_bounded_by_k_or_rerank_n() -> None:
    rig = build(rerank=False, k=2)
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert result.ranking == [h.chunk_id for h in result.hits]
    assert len(result.ranking) == 2
    rig = build(rerank=True, k=1, rerank_n=2)
    result = await rig.search("beta", mode=SearchMode.VECTOR)
    assert len(result.ranking) == 2


async def test_search_ranked_documents_follow_the_ranking() -> None:
    rig = build(rerank=True, rerank_n=2, k=1)
    result = await rig.search("beta", mode=SearchMode.HYBRID, depth=4)
    by_chunk = {c.id: c.document_id for c in rig.collection.chunks}
    assert len(result.ranked_documents) == len(result.ranking) == 4
    assert result.ranked_documents == [by_chunk[c] for c in result.ranking]


async def test_search_overrides_reach_the_lexical_leg() -> None:
    rig = build(rerank=False)
    overrides = SearchOverrides(
        lexical_match=LexicalMatch.WEBSEARCH, lexical_rank=LexicalRank.TS_RANK_CD
    )
    await rig.search("beta", mode=SearchMode.LEXICAL, overrides=overrides)
    await rig.search("beta", mode=SearchMode.LEXICAL)
    first, second = rig.store.lexical_queries
    assert (first.match, first.rank) == (LexicalMatch.WEBSEARCH, LexicalRank.TS_RANK_CD)
    assert (second.match, second.rank) == (LexicalMatch.ANY, LexicalRank.TS_RANK)


async def test_search_override_weights_change_fusion_and_keep_other_legs() -> None:
    rig = build(rerank=False, rrf_weights={"lexical": 0.3, "vector": 1.0})
    lexical_only = SearchOverrides(rrf_weights={"lexical": 1.0})
    result = await rig.search("beta", mode=SearchMode.HYBRID, overrides=lexical_only)
    assert result.collection.settings.retrieval.rrf_weights == {"lexical": 1.0, "vector": 1.0}
    hit = result.hits[0]
    expected = sum(1.0 / (60 + rank) for rank in hit.ranks.values())
    assert hit.rrf_score == pytest.approx(expected)
    default = await rig.search("beta", mode=SearchMode.HYBRID)
    assert default.collection.settings.retrieval.rrf_weights == {"lexical": 0.3, "vector": 1.0}


async def test_search_override_reranker_and_rerank_n_replace_the_service_ones() -> None:
    rig = build(rerank=True, rerank_n=2)
    other = FakeReranker()
    overrides = SearchOverrides(rerank_n=3, reranker=other)
    result = await rig.search("beta", mode=SearchMode.HYBRID, overrides=overrides, depth=4)
    assert result.reranked
    assert rig.reranker is not None
    assert rig.reranker.calls == []
    assert len(other.calls) == 1
    assert len(other.calls[0][1]) == 3


async def test_search_override_reranker_enables_rerank_without_a_service_reranker() -> None:
    rig = build(rerank=True, with_reranker=False)
    plain = await rig.search("beta", mode=SearchMode.HYBRID)
    assert not plain.reranked
    other = FakeReranker()
    result = await rig.search(
        "beta", mode=SearchMode.HYBRID, overrides=SearchOverrides(reranker=other)
    )
    assert result.reranked
    assert other.calls


async def test_search_overrides_do_not_rerank_when_the_request_says_no() -> None:
    rig = build(rerank=True)
    other = FakeReranker()
    result = await rig.search(
        "beta", mode=SearchMode.HYBRID, rerank=False, overrides=SearchOverrides(reranker=other)
    )
    assert not result.reranked
    assert other.calls == []


async def test_search_leg_limit_raises_leg_sizes() -> None:
    rig = build(rerank=False, k_lexical=5)
    await rig.search("beta", mode=SearchMode.LEXICAL, leg_limit=40)
    await rig.search("beta", mode=SearchMode.LEXICAL)
    assert rig.store.lexical_limits == [40, 5]


async def test_search_stage_seconds_cover_every_stage() -> None:
    rig = build(rerank=True)
    result = await rig.search("beta", mode=SearchMode.HYBRID)
    assert set(result.stage_seconds) == {"embed_query", "lexical", "vector", "rerank"}
    assert all(v == pytest.approx(0.5) for v in result.stage_seconds.values())


async def test_search_reranker_wrong_count_raises() -> None:
    rig = build(rerank=True)
    assert rig.reranker is not None
    rig.reranker.wrong_count = True
    with pytest.raises(RuntimeError, match="scores for"):
        await rig.search("beta", mode=SearchMode.LEXICAL)


async def test_search_runs_legs_inside_snapshot_and_rerank_after_exit() -> None:
    rig = build(rerank=True)
    await rig.search("beta", mode=SearchMode.HYBRID, need_similarity=True)
    assert rig.events == [
        "enter",
        "collection",
        "lexical",
        "vector",
        "hits",
        "similarities",
        "exit",
        "rerank",
    ]


async def test_search_exit_recorded_when_validation_fails_inside_snapshot() -> None:
    rig = build()
    with pytest.raises(ValidationError):
        await rig.search("beta", k=0)
    assert rig.events == ["enter", "collection", "exit"]


async def test_search_result_exposes_collection_and_space() -> None:
    rig = build(rerank=False)
    result = await rig.search("beta", mode=SearchMode.LEXICAL)
    assert result.space == rig.collection.space
    assert result.collection.chunk_count == 4
    assert result.collection.settings.retrieval.k == 8
