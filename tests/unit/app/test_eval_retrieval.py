# SPDX-License-Identifier: Apache-2.0
"""`eval retrieval` runner: ablation row plans, requests (depth/legs), pooling and scoring."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest

from footnote.app.evaluation.datasets import BeirDataset, CorpusDoc
from footnote.app.evaluation.retrieval import (
    EVAL_DEPTH,
    EvalError,
    EvalMode,
    IndexedCollection,
    RunOutput,
    Shipped,
    Variant,
    make_row,
    parse_chunkings,
    parse_modes,
    plan_ablation,
    plan_variants,
    run_variant,
    split_queries,
)
from footnote.app.search import SearchService
from footnote.domain.documents import CollectionId, DocumentId
from footnote.domain.settings import (
    ChunkingStrategyName,
    CollectionSettings,
    LexicalMatch,
    LexicalRank,
    RetrievalSettings,
    SearchMode,
)
from footnote.domain.tenancy import TenantId

from .fakes import FakeClock, FakeCollection, FakeEmbedderFactory, FakeReranker, FakeSearchStore
from .fakes import make_space as fake_space

HEADING = ChunkingStrategyName.HEADING
RECURSIVE = ChunkingStrategyName.RECURSIVE
SHIPPED = Shipped(RetrievalSettings(), HEADING, "int8")
TENANT = TenantId(UUID(int=7))


def test_parse_modes_and_chunkings() -> None:
    assert parse_modes("vector, hybrid+rerank,vector") == [EvalMode.VECTOR, EvalMode.HYBRID_RERANK]
    assert parse_chunkings("heading,recursive") == [HEADING, RECURSIVE]
    with pytest.raises(EvalError, match="unknown mode"):
        parse_modes("vector,bm25")
    with pytest.raises(EvalError, match="unknown chunking"):
        parse_chunkings("semantic")


def test_plan_variants_is_modes_times_chunkings_at_shipped_settings() -> None:
    variants = plan_variants(SHIPPED, list(EvalMode), [HEADING, RECURSIVE], "train")
    assert len(variants) == 8
    assert {v.split for v in variants} == {"train"}
    assert all(v.lexical_weight == pytest.approx(0.2) for v in variants)
    shipped = [v for v in variants if v.is_shipped(SHIPPED)]
    assert {v.chunking for v in shipped} == {HEADING}
    assert len(shipped) == 4


def test_plan_ablation_covers_every_m3_row() -> None:
    rows = plan_ablation(SHIPPED)
    keys = {(v.mode, v.split, v.chunking) for v in rows}
    assert (EvalMode.VECTOR, "test", HEADING) in keys
    lexical = {(v.lexical_match, v.lexical_rank) for v in rows if v.mode is EvalMode.LEXICAL}
    assert lexical == {
        (LexicalMatch.ANY, LexicalRank.TS_RANK),
        (LexicalMatch.ANY, LexicalRank.TS_RANK_CD),
        (LexicalMatch.ALL, LexicalRank.TS_RANK),
        (LexicalMatch.WEBSEARCH, LexicalRank.TS_RANK),
    }
    hybrid = {(v.split, v.lexical_weight) for v in rows if v.mode is EvalMode.HYBRID}
    assert hybrid == {(s, w) for s in ("train", "test") for w in (1.0, 0.5, 0.3, 0.2)}
    reranked = [v for v in rows if v.mode is EvalMode.HYBRID_RERANK]
    on_heading = {(v.reranker_variant, v.rerank_n) for v in reranked if v.chunking == HEADING}
    assert on_heading == {("int8", 30), ("int8", 50), ("fp32", 30), ("fp32", 50)}
    on_recursive = [v for v in reranked if v.chunking == RECURSIVE]
    assert [(v.reranker_variant, v.rerank_n) for v in on_recursive] == [("int8", 30)]
    assert len(rows) == 1 + 4 + 8 + 4 + 1
    marked = [v.name for v in rows if v.is_shipped(SHIPPED)]
    assert marked == [
        "vector [heading]",
        "lexical any+ts_rank [heading]",
        "hybrid lexical weight 0.2 [heading]",
        "hybrid lexical weight 0.2 [heading]",
        "hybrid+rerank int8 rerank_n 30 [heading]",
    ]


def test_plan_ablation_adds_a_shipped_value_outside_the_sweep() -> None:
    retrieval = replace(RetrievalSettings(), rrf_weights={"lexical": 0.25, "vector": 1.0})
    rows = plan_ablation(Shipped(retrieval, RECURSIVE, "fp32"))
    weights = {v.lexical_weight for v in rows if v.mode is EvalMode.HYBRID}
    assert weights == {1.0, 0.5, 0.3, 0.25, 0.2}
    chunkings = {v.chunking for v in rows if v.mode is EvalMode.HYBRID_RERANK}
    assert chunkings == {HEADING, RECURSIVE}


def test_variant_request_uses_eval_depth_legs_and_overrides() -> None:
    reranker = FakeReranker()
    variant = Variant.shipped(SHIPPED, EvalMode.HYBRID_RERANK, "test", HEADING).with_(
        reranker_variant="fp32", rerank_n=50, lexical_weight=0.5
    )
    request = variant.request("q", {"fp32": reranker})
    assert (request.mode, request.rerank) == (SearchMode.HYBRID, True)
    assert (request.depth, request.leg_limit) == (EVAL_DEPTH, EVAL_DEPTH)
    assert request.overrides is not None
    assert request.overrides.reranker is reranker
    assert request.overrides.rerank_n == 50
    assert request.overrides.rrf_weights == {"lexical": 0.5}
    with pytest.raises(EvalError, match="no int8 reranker"):
        variant.with_(reranker_variant="int8").request("q", {"fp32": reranker})
    lexical = Variant.shipped(SHIPPED, EvalMode.LEXICAL, "test", HEADING).request("q", {})
    assert (lexical.mode, lexical.rerank) == (SearchMode.LEXICAL, False)


def test_variant_settings_describe_only_what_the_mode_uses() -> None:
    retrieval = RetrievalSettings()
    vector = Variant.shipped(SHIPPED, EvalMode.VECTOR, "test", HEADING)
    assert vector.settings(retrieval) == "chunking=heading"
    full = Variant.shipped(SHIPPED, EvalMode.HYBRID_RERANK, "test", HEADING)
    assert full.settings(retrieval) == (
        "chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 "
        "rrf_weights=lexical:0.2,vector:1 reranker=int8 rerank_n=30"
    )
    # vector rows ignore lexical settings when deciding "shipped"
    assert vector.with_(lexical_weight=1.0).is_shipped(SHIPPED)


def _service(
    texts: list[str], *, rerank_n: int = 2, docs: list[int] | None = None
) -> tuple[SearchService, IndexedCollection, FakeReranker]:
    settings = CollectionSettings(retrieval=replace(RetrievalSettings(), rerank_n=rerank_n))
    collection = FakeCollection(fake_space(), settings)
    for text in texts:
        collection.add(text)
    mapping: dict[DocumentId, str] = {}
    for n, chunk in enumerate(collection.chunks):
        doc = DocumentId(UUID(int=5000 + (docs[n] if docs else n)))
        chunk.document_id = doc
        mapping[doc] = f"d{docs[n] if docs else n}"
    reranker = FakeReranker()
    service = SearchService(FakeSearchStore(collection), FakeEmbedderFactory(), None, FakeClock())
    target = IndexedCollection(TENANT, CollectionId(UUID(int=99)), mapping)
    return service, target, reranker


async def test_run_variant_pools_chunks_to_documents_by_first_occurrence() -> None:
    texts = ["beta one", "beta two", "beta three", "gamma"]
    service, target, _ = _service(texts, docs=[0, 0, 1, 2])
    variant = Variant.shipped(SHIPPED, EvalMode.LEXICAL, "test", HEADING)
    ticks = iter([0.0, 0.25, 1.0, 1.5])
    output = await run_variant(
        service, target, [("q1", "beta"), ("q2", "gamma")], variant, {}, clock=lambda: next(ticks)
    )
    assert output.rankings == {"q1": ["d0", "d1"], "q2": ["d2"]}
    assert output.seconds == {"q1": 0.25, "q2": 0.5}


async def test_run_variant_rerank_head_then_rrf_remainder() -> None:
    # The fake reranker scores by query-word overlap: "beta gamma" ranks chunk 3 first.
    texts = ["beta", "beta delta", "beta gamma", "beta epsilon"]
    service, target, reranker = _service(texts, rerank_n=3)
    hybrid = Variant.shipped(SHIPPED, EvalMode.HYBRID, "test", HEADING)
    plain = await run_variant(service, target, [("q", "beta gamma")], hybrid, {})
    reranked = hybrid.with_(mode=EvalMode.HYBRID_RERANK, rerank_n=2)
    output = await run_variant(service, target, [("q", "beta gamma")], reranked, {"int8": reranker})
    head = output.rankings["q"][:2]
    assert head[0] == "d2"
    assert set(head) <= set(plain.rankings["q"][:2]) | {"d2"}
    assert output.rankings["q"][2:] == [d for d in plain.rankings["q"] if d not in head]
    assert len(reranker.calls[0][1]) == 2


async def test_run_variant_fails_when_rerank_did_not_happen() -> None:
    service, target, _ = _service(["beta"])
    variant = Variant.shipped(SHIPPED, EvalMode.HYBRID_RERANK, "test", HEADING)
    with pytest.raises(EvalError, match="no int8 reranker"):
        await run_variant(service, target, [("q", "beta")], variant, {})


def _dataset() -> BeirDataset:
    corpus = {i: CorpusDoc(i, f"T{i}", "x") for i in ("d0", "d1", "d2")}
    queries = {"q1": "one", "q2": "two", "q3": "three"}
    qrels = {"test": {"q1": {"d0": 1}, "q2": {"d1": 1, "d2": 0}, "q9": {"d0": 1}}}
    return BeirDataset("toy", corpus, queries, qrels)


def test_split_queries_keeps_qrels_order_and_known_queries() -> None:
    assert split_queries(_dataset(), "test") == [("q1", "one"), ("q2", "two")]
    with pytest.raises(EvalError, match="no qrels split 'train'"):
        split_queries(_dataset(), "train")


def test_make_row_scores_judged_queries_and_latency_percentiles() -> None:
    output = RunOutput({"q1": ["d0", "d1"], "q2": ["d0", "d2", "d1"]}, {"q1": 0.010, "q2": 0.030})
    row = make_row(
        "toy",
        split="test",
        mode="lexical",
        chunking="heading",
        settings="s",
        output=output,
        qrels=_dataset().qrels["test"],
        shipped=True,
    )
    assert row.queries == 2
    assert row.metrics["recall_at_1"] == pytest.approx(0.5)
    assert row.metrics["mrr_at_10"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert row.p50_ms == pytest.approx(20.0)
    assert row.p95_ms == pytest.approx(29.0)
    assert row.shipped
    assert row.group == "lexical"
    with pytest.raises(EvalError, match="no judged queries"):
        make_row(
            "none",
            split="test",
            mode="m",
            chunking="c",
            settings="s",
            output=RunOutput({"qx": []}, {"qx": 0.0}),
            qrels=_dataset().qrels["test"],
        )
