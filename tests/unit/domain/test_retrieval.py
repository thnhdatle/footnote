# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion, the lexical query plan, pooling and vector-path selection."""

from __future__ import annotations

from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.documents import ChunkId
from footnote.domain.retrieval import (
    FusionConfig,
    LegHit,
    RankedList,
    ef_search,
    eval_order,
    lexical_rank_sql,
    lexical_tsquery_sql,
    pool_documents,
    rrf,
    select_vector_path,
)
from footnote.domain.settings import Language, LexicalMatch, LexicalRank

A = ChunkId(UUID(int=1))
B = ChunkId(UUID(int=2))
C = ChunkId(UUID(int=3))


def _leg(name: str, *hits: tuple[ChunkId, float]) -> RankedList:
    return RankedList(name, tuple(LegHit(c, s) for c, s in hits))


def test_fusion_config_defaults() -> None:
    config = FusionConfig()
    assert config.rrf_k == 60
    assert dict(config.weights) == {"lexical": 0.2, "vector": 1.0}


def test_rrf_weighted_two_legs_exact_scores_and_order() -> None:
    lexical = _leg("lexical", (A, 2.0), (B, 1.0))
    vector = _leg("vector", (B, 0.9), (C, 0.8))
    fused = rrf([lexical, vector], FusionConfig())
    assert [h.chunk_id for h in fused] == [B, C, A]
    assert fused[0].rrf_score == pytest.approx(0.2 / 62 + 1.0 / 61)
    assert fused[1].rrf_score == pytest.approx(1.0 / 62)
    assert fused[2].rrf_score == pytest.approx(0.2 / 61)
    assert fused[0].ranks == {"lexical": 2, "vector": 1}
    assert fused[0].scores == {"lexical": 1.0, "vector": 0.9}
    assert fused[1].ranks == {"vector": 2}
    assert fused[2].scores == {"lexical": 2.0}


def test_rrf_uses_configured_k() -> None:
    config = FusionConfig(rrf_k=10, weights={"vector": 2.0})
    fused = rrf([_leg("vector", (A, 0.5), (B, 0.4))], config)
    assert fused[0].rrf_score == pytest.approx(2.0 / 11)
    assert fused[1].rrf_score == pytest.approx(2.0 / 12)


def test_rrf_duplicate_in_leg_counts_first_rank_only() -> None:
    fused = rrf([_leg("vector", (A, 0.9), (B, 0.8), (A, 0.7))], FusionConfig())
    assert [h.chunk_id for h in fused] == [A, B]
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused[0].ranks == {"vector": 1}
    assert fused[0].scores == {"vector": 0.9}


def test_rrf_zero_weight_leg_records_ranks_but_adds_nothing() -> None:
    config = FusionConfig(weights={"lexical": 0.0, "vector": 1.0})
    fused = rrf([_leg("lexical", (A, 3.0)), _leg("vector", (B, 0.5))], config)
    assert [h.chunk_id for h in fused] == [B, A]
    assert fused[1].rrf_score == 0.0
    assert fused[1].ranks == {"lexical": 1}
    assert fused[1].scores == {"lexical": 3.0}


def test_rrf_missing_weight_raises() -> None:
    config = FusionConfig(weights={"vector": 1.0})
    with pytest.raises(ValueError, match="lexical"):
        rrf([_leg("lexical", (A, 1.0))], config)


def test_rrf_empty_input() -> None:
    assert rrf([], FusionConfig()) == []


def test_rrf_tie_prefers_chunk_present_in_vector_leg() -> None:
    config = FusionConfig(weights={"lexical": 1.0, "vector": 1.0})
    fused = rrf([_leg("lexical", (A, 1.0)), _leg("vector", (B, 0.1))], config)
    assert [h.chunk_id for h in fused] == [B, A]


def test_rrf_tie_prefers_higher_vector_score() -> None:
    config = FusionConfig(weights={"lexical": 1.0, "vector": 1.0})
    lexical = _leg("lexical", (A, 1.0), (B, 0.5))
    vector = _leg("vector", (B, 0.9), (A, 0.5))
    fused = rrf([lexical, vector], config)
    assert fused[0].rrf_score == fused[1].rrf_score
    assert [h.chunk_id for h in fused] == [B, A]


def test_rrf_tie_falls_back_to_chunk_id() -> None:
    config = FusionConfig(weights={"lexical": 1.0, "vector": 1.0})
    lexical = _leg("lexical", (B, 1.0), (A, 0.5))
    vector = _leg("vector", (A, 0.5), (B, 0.5))
    fused = rrf([lexical, vector], config)
    assert [h.chunk_id for h in fused] == [A, B]


def test_rrf_tie_without_vector_scores_orders_by_chunk_id() -> None:
    config = FusionConfig(weights={"lexical": 1.0, "other": 1.0})
    fused = rrf([_leg("lexical", (C, 1.0)), _leg("other", (A, 1.0))], config)
    assert [h.chunk_id for h in fused] == [A, C]


_chunk_ids = st.integers(min_value=0, max_value=20).map(lambda n: ChunkId(UUID(int=n)))
_hits = st.lists(st.tuples(_chunk_ids, st.floats(0, 1)), max_size=15)


@given(lexical=_hits, vector=_hits)
def test_rrf_output_is_permutation_of_input_union(
    lexical: list[tuple[ChunkId, float]], vector: list[tuple[ChunkId, float]]
) -> None:
    fused = rrf([_leg("lexical", *lexical), _leg("vector", *vector)], FusionConfig())
    ids = [h.chunk_id for h in fused]
    assert len(ids) == len(set(ids))
    assert set(ids) == {c for c, _ in lexical} | {c for c, _ in vector}
    scores = [h.rrf_score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_lexical_tsquery_any_exact_sql() -> None:
    assert lexical_tsquery_sql(LexicalMatch.ANY, Language.ENGLISH) == (
        "(SELECT to_tsquery('simple', string_agg(quote_literal(lex), ' | ' ORDER BY lex)) "
        "FROM unnest(tsvector_to_array(to_tsvector('english'::regconfig, %(query)s))) AS lex)"
    )


def test_lexical_tsquery_all_exact_sql() -> None:
    assert lexical_tsquery_sql(LexicalMatch.ALL, Language.SIMPLE) == (
        "plainto_tsquery('simple'::regconfig, %(query)s)"
    )


def test_lexical_tsquery_websearch_exact_sql() -> None:
    assert lexical_tsquery_sql(LexicalMatch.WEBSEARCH, Language.ENGLISH) == (
        "websearch_to_tsquery('english'::regconfig, %(query)s)"
    )


@pytest.mark.parametrize("match", list(LexicalMatch))
@pytest.mark.parametrize("language", list(Language))
def test_lexical_tsquery_only_uses_query_placeholder(
    match: LexicalMatch, language: Language
) -> None:
    sql = lexical_tsquery_sql(match, language)
    assert "%(query)s" in sql
    assert sql.replace("%(query)s", "").count("%") == 0
    assert "{" not in sql
    assert f"'{language.value}'" in sql


def test_lexical_rank_sql_default_alias() -> None:
    assert lexical_rank_sql(LexicalRank.TS_RANK) == "ts_rank(c.tsv, q)"
    assert lexical_rank_sql(LexicalRank.TS_RANK_CD) == "ts_rank_cd(c.tsv, q)"


def test_lexical_rank_sql_custom_alias() -> None:
    assert lexical_rank_sql(LexicalRank.TS_RANK, "_tq1") == "ts_rank(c.tsv, _tq1)"


@pytest.mark.parametrize("alias", ["", "Q", "1q", "q; DROP", "q-1", "q\n", "q "])
def test_lexical_rank_sql_rejects_bad_alias(alias: str) -> None:
    with pytest.raises(ValueError, match="alias"):
        lexical_rank_sql(LexicalRank.TS_RANK, alias)


def test_pool_documents_keeps_first_occurrence_order() -> None:
    assert pool_documents(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert pool_documents(iter([])) == []


@given(st.lists(st.integers()))
def test_pool_documents_is_distinct_and_covers_input(ids: list[int]) -> None:
    pooled = pool_documents(ids)
    assert len(pooled) == len(set(pooled))
    assert set(pooled) == set(ids)


def test_eval_order_without_rerank_is_fused() -> None:
    assert eval_order([A, B, C], None, 2) == [A, B, C]


def test_eval_order_rerank_head_then_remaining_fused() -> None:
    assert eval_order([A, B, C], [C, A], 5) == [C, A, B]


def test_eval_order_truncates_rerank_to_rerank_n() -> None:
    assert eval_order([A, B, C], [C, B, A], 1) == [C, A, B]


def test_eval_order_empty_rerank_keeps_fused() -> None:
    assert eval_order([A, B], [], 3) == [A, B]


def test_select_vector_path_boundary() -> None:
    assert select_vector_path(100, 100) == "exact"
    assert select_vector_path(101, 100) == "hnsw"
    assert select_vector_path(0, 0) == "exact"


def test_ef_search_is_max_of_both() -> None:
    assert ef_search(40, 100) == 100
    assert ef_search(200, 100) == 200
    assert ef_search(50, 50) == 50
