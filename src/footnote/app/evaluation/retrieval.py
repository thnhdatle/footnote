# SPDX-License-Identifier: Apache-2.0
"""`footnote eval retrieval`: index a BEIR dataset into a temporary tenant,
run every query of a split per configuration, pool chunks to documents and score them.

Depth and pooling: every search runs with the internal `depth` 100 and leg sizes
`k_lexical` = `k_vector` = max(configured, 100) (`SearchRequest.leg_limit`); with rerank
the top `rerank_n` come in rerank order, then the remaining hits in RRF order
(`domain.retrieval.eval_order`). Chunks are pooled to their BEIR documents by first
occurrence (`pool_documents`) and the metrics cut that list. Settings swept by the
ablation (lexical match/rank, RRF weights, reranker file, `rerank_n`) reach the search as
`SearchOverrides` over one indexed collection; a chunking strategy is its own collection.
Latency is the in-process `SearchService.search` time (query embedding included).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from footnote.app.admin import AdminService
from footnote.app.collections import CollectionService, SettingsPatch
from footnote.app.evaluation.bm25 import Bm25Index, fuse_rankings
from footnote.app.evaluation.corpus import IndexedCorpus, Uploader, index_corpus
from footnote.app.evaluation.datasets import BeirDataset, query_ids
from footnote.app.evaluation.metrics import evaluate_run, percentile
from footnote.app.evaluation.report import Row
from footnote.app.ingest import IngestService
from footnote.app.search import SearchOverrides, SearchRequest, SearchResult
from footnote.domain.documents import CollectionId, DocumentId
from footnote.domain.ports import Reranker
from footnote.domain.retrieval import pool_documents
from footnote.domain.settings import (
    ChunkingStrategyName,
    Language,
    LexicalMatch,
    LexicalRank,
    RetrievalSettings,
    SearchMode,
)
from footnote.domain.tenancy import TenantId

EVAL_DEPTH = 100
LEXICAL_SWEEP = (
    (LexicalMatch.ANY, LexicalRank.TS_RANK),
    (LexicalMatch.ANY, LexicalRank.TS_RANK_CD),
    (LexicalMatch.ALL, LexicalRank.TS_RANK),
    (LexicalMatch.WEBSEARCH, LexicalRank.TS_RANK),
)
WEIGHT_SWEEP = (1.0, 0.5, 0.3, 0.2)
RERANKER_SWEEP = ("int8", "fp32")
RERANK_N_SWEEP = (30, 50)
ABLATION_SPLITS = ("train", "test")
TEST = "test"


class EvalMode(StrEnum):
    """Evaluation modes: one leg, both fused, or fused and reranked."""

    VECTOR = "vector"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


class EvalError(RuntimeError):
    """The evaluation cannot run as asked (bad option, missing split or reranker)."""


def parse_modes(text: str) -> list[EvalMode]:
    """Comma-separated modes, in the given order, deduplicated."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    try:
        return list(dict.fromkeys(EvalMode(n) for n in names))
    except ValueError as exc:
        allowed = ", ".join(m.value for m in EvalMode)
        raise EvalError(f"unknown mode in {text!r}; allowed: {allowed}") from exc


def parse_chunkings(text: str) -> list[ChunkingStrategyName]:
    """Comma-separated chunking strategies, deduplicated."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    try:
        return list(dict.fromkeys(ChunkingStrategyName(n) for n in names))
    except ValueError as exc:
        allowed = ", ".join(c.value for c in ChunkingStrategyName)
        raise EvalError(f"unknown chunking in {text!r}; allowed: {allowed}") from exc


@dataclass(frozen=True)
class Shipped:
    """The shipped defaults the rows vary from (config collection defaults)."""

    retrieval: RetrievalSettings
    chunking: ChunkingStrategyName
    reranker_variant: str

    @property
    def lexical_weight(self) -> float:
        """Shipped `rrf_weights.lexical`."""
        return float(self.retrieval.rrf_weights.get("lexical", 0.0))


@dataclass(frozen=True)
class Variant:
    """One fully resolved configuration on one split."""

    mode: EvalMode
    split: str
    chunking: ChunkingStrategyName
    lexical_match: LexicalMatch
    lexical_rank: LexicalRank
    lexical_weight: float
    reranker_variant: str
    rerank_n: int
    group: str = ""

    @classmethod
    def shipped(
        cls, shipped: Shipped, mode: EvalMode, split: str, chunking: ChunkingStrategyName
    ) -> Variant:
        """`mode` with every setting at its shipped value."""
        r = shipped.retrieval
        return cls(
            mode,
            split,
            chunking,
            r.lexical_match,
            r.lexical_rank,
            shipped.lexical_weight,
            shipped.reranker_variant,
            r.rerank_n,
            group=mode.value,
        )

    def with_(self, **changes: object) -> Variant:
        """A copy with `changes`."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def is_shipped(self, shipped: Shipped) -> bool:
        """Every setting this mode uses is at its shipped value."""
        base = Variant.shipped(shipped, self.mode, self.split, shipped.chunking)
        return self.key() == base.key()

    def key(self) -> tuple[object, ...]:
        """Identity over the settings the mode actually uses."""
        lexical = (self.lexical_match, self.lexical_rank)
        fused = (self.lexical_weight,)
        rerank = (self.reranker_variant, self.rerank_n)
        parts: dict[EvalMode, tuple[object, ...]] = {
            EvalMode.VECTOR: (),
            EvalMode.LEXICAL: lexical,
            EvalMode.HYBRID: lexical + fused,
            EvalMode.HYBRID_RERANK: lexical + fused + rerank,
        }
        return (self.mode, self.split, self.chunking, *parts[self.mode])

    @property
    def name(self) -> str:
        """Row label."""
        labels = {
            EvalMode.VECTOR: "vector",
            EvalMode.LEXICAL: f"lexical {self.lexical_match}+{self.lexical_rank}",
            EvalMode.HYBRID: f"hybrid lexical weight {self.lexical_weight:g}",
            EvalMode.HYBRID_RERANK: (
                f"hybrid+rerank {self.reranker_variant} rerank_n {self.rerank_n}"
            ),
        }
        return f"{labels[self.mode]} [{self.chunking}]"

    def settings(self, retrieval: RetrievalSettings) -> str:
        """The settings this row runs with (CSV `settings` column)."""
        parts = [f"chunking={self.chunking}"]
        if self.mode in (EvalMode.LEXICAL, EvalMode.HYBRID, EvalMode.HYBRID_RERANK):
            parts.append(f"lexical_match={self.lexical_match} lexical_rank={self.lexical_rank}")
        if self.mode in (EvalMode.HYBRID, EvalMode.HYBRID_RERANK):
            vector = retrieval.rrf_weights.get("vector", 1.0)
            parts.append(
                f"rrf_k={retrieval.rrf_k} rrf_weights=lexical:{self.lexical_weight:g}"
                f",vector:{vector:g}"
            )
        if self.mode is EvalMode.HYBRID_RERANK:
            parts.append(f"reranker={self.reranker_variant} rerank_n={self.rerank_n}")
        return " ".join(parts)

    def request(self, query: str, rerankers: Mapping[str, Reranker]) -> SearchRequest:
        """The app-layer search request of this configuration (depth 100, legs ≥ 100)."""
        rerank = self.mode is EvalMode.HYBRID_RERANK
        reranker: Reranker | None = None
        if rerank:
            reranker = rerankers.get(self.reranker_variant)
            if reranker is None:
                raise EvalError(f"no {self.reranker_variant} reranker configured for {self.name}")
        mode = {
            EvalMode.VECTOR: SearchMode.VECTOR,
            EvalMode.LEXICAL: SearchMode.LEXICAL,
            EvalMode.HYBRID: SearchMode.HYBRID,
            EvalMode.HYBRID_RERANK: SearchMode.HYBRID,
        }[self.mode]
        return SearchRequest(
            query,
            mode=mode,
            rerank=rerank,
            depth=EVAL_DEPTH,
            leg_limit=EVAL_DEPTH,
            overrides=SearchOverrides(
                lexical_match=self.lexical_match,
                lexical_rank=self.lexical_rank,
                rrf_weights={"lexical": self.lexical_weight},
                rerank_n=self.rerank_n,
                reranker=reranker,
            ),
        )


def _unique(variants: Sequence[Variant]) -> list[Variant]:
    seen: set[tuple[object, ...]] = set()
    out: list[Variant] = []
    for v in variants:
        if v.key() not in seen:
            seen.add(v.key())
            out.append(v)
    return out


def plan_variants(
    shipped: Shipped,
    modes: Sequence[EvalMode],
    chunkings: Sequence[ChunkingStrategyName],
    split: str,
) -> list[Variant]:
    """Without `--ablation`: every mode and chunking at shipped settings on `split`."""
    return _unique([Variant.shipped(shipped, m, split, c) for c in chunkings for m in modes])


def plan_ablation(shipped: Shipped) -> list[Variant]:
    """The ablation row set, shipped values always included and marked by the report."""
    c = shipped.chunking
    vector = Variant.shipped(shipped, EvalMode.VECTOR, TEST, c)
    lexical = Variant.shipped(shipped, EvalMode.LEXICAL, TEST, c)
    hybrid = Variant.shipped(shipped, EvalMode.HYBRID, TEST, c)
    reranked = Variant.shipped(shipped, EvalMode.HYBRID_RERANK, TEST, c)
    rows = [vector]
    pairs = [*LEXICAL_SWEEP, (lexical.lexical_match, lexical.lexical_rank)]
    rows += [lexical.with_(lexical_match=m, lexical_rank=r) for m, r in pairs]
    weights = sorted({*WEIGHT_SWEEP, shipped.lexical_weight}, reverse=True)
    rows += [
        hybrid.with_(split=s, lexical_weight=w, group="hybrid weight sweep")
        for s in ABLATION_SPLITS
        for w in weights
    ]
    variants = list(dict.fromkeys([*RERANKER_SWEEP, shipped.reranker_variant]))
    sizes = sorted({*RERANK_N_SWEEP, shipped.retrieval.rerank_n})
    rows += [reranked.with_(reranker_variant=v, rerank_n=n) for v in variants for n in sizes]
    for other in (ChunkingStrategyName.HEADING, ChunkingStrategyName.RECURSIVE):
        rows.append(reranked.with_(chunking=other, group="chunking"))
    return _unique(rows)


class Searcher(Protocol):
    """`SearchService.search`."""

    async def search(
        self, tenant: TenantId, collection_id: CollectionId, request: SearchRequest
    ) -> SearchResult:
        """One search."""


@dataclass(frozen=True)
class IndexedCollection:
    """A collection of the temporary tenant holding the corpus with one chunking."""

    tenant: TenantId
    collection_id: CollectionId
    doc_ids: Mapping[DocumentId, str]


@dataclass(frozen=True)
class RunOutput:
    """Pooled document rankings (top `EVAL_DEPTH`) and search seconds per query."""

    rankings: dict[str, list[str]]
    seconds: dict[str, float]


def pooled(result: SearchResult, doc_ids: Mapping[DocumentId, str]) -> list[str]:
    """BEIR document ids of a result's evaluation ranking, first occurrence, top 100."""
    return pool_documents(doc_ids[d] for d in result.ranked_documents)[:EVAL_DEPTH]


async def run_variant(  # noqa: PLR0913 - the run inputs
    search: Searcher,
    target: IndexedCollection,
    queries: Sequence[tuple[str, str]],
    variant: Variant,
    rerankers: Mapping[str, Reranker],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> RunOutput:
    """Run every query sequentially (latency is one query at a time)."""
    rankings: dict[str, list[str]] = {}
    seconds: dict[str, float] = {}
    for qid, text in queries:
        request = variant.request(text, rerankers)
        start = clock()
        result = await search.search(target.tenant, target.collection_id, request)
        seconds[qid] = clock() - start
        if variant.mode is EvalMode.HYBRID_RERANK and not result.reranked:
            raise EvalError(f"{variant.name}: the search did not rerank")
        rankings[qid] = pooled(result, target.doc_ids)
    return RunOutput(rankings, seconds)


def split_queries(dataset: BeirDataset, split: str) -> list[tuple[str, str]]:
    """(query id, text) of `split` in qrels order; EvalError for an unknown split."""
    if split not in dataset.qrels:
        known = ", ".join(sorted(dataset.qrels)) or "none"
        raise EvalError(f"dataset {dataset.name} has no qrels split {split!r} (has: {known})")
    return [(q, dataset.queries[q]) for q in query_ids(dataset, split) if q in dataset.queries]


def make_row(  # noqa: PLR0913 - one value per report column group
    name: str,
    *,
    split: str,
    mode: str,
    chunking: str,
    settings: str,
    output: RunOutput,
    qrels: Mapping[str, Mapping[str, int]],
    shipped: bool = False,
    group: str = "",
    note: str = "",
) -> Row:
    """Score a run against the qrels of its queries (seconds → ms percentiles)."""
    judged = {q: qrels[q] for q in output.rankings if q in qrels}
    if not judged:
        raise EvalError(f"{name}: no judged queries")
    ms = [s * 1000.0 for s in output.seconds.values()]
    return Row(
        name=name,
        split=split,
        mode=mode,
        chunking=chunking,
        settings=settings,
        metrics=evaluate_run(output.rankings, judged),
        p50_ms=percentile(ms, 50),
        p95_ms=percentile(ms, 95),
        queries=len(judged),
        shipped=shipped,
        group=group or mode,
        note=note,
    )


# --- orchestration ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvalServices:
    """App services of the in-process runtime the harness drives."""

    admin: AdminService
    collections: CollectionService
    uploader: Uploader
    ingest: IngestService
    search: Searcher
    rerankers: Mapping[str, Reranker]
    delete_tenant: Callable[[TenantId], Awaitable[None]]


type Log = Callable[[str], None]
type CorpusHook = Callable[[ChunkingStrategyName, IndexedCorpus], None]


def _quiet(_message: str) -> None:
    return None


@dataclass(frozen=True)
class EvalOptions:
    """How to run: the variants, tenant handling, ingest concurrency, baselines."""

    variants: Sequence[Variant]
    shipped: Shipped
    run_id: str
    keep: bool = False
    concurrency: int = 8
    bm25: bool = True
    limit: int | None = None
    log: Log = _quiet
    on_indexed: CorpusHook | None = None

    def queries(self, dataset: BeirDataset, split: str) -> list[tuple[str, str]]:
        """The split's queries, the first `limit` only when set (`make eval-quick`)."""
        queries = split_queries(dataset, split)
        return queries if self.limit is None else queries[: self.limit]


@dataclass
class EvalOutcome:
    """Rows in report order plus what was indexed."""

    rows: list[Row]
    tenant: TenantId
    kept: bool
    corpora: dict[ChunkingStrategyName, IndexedCorpus] = field(default_factory=dict)
    seconds: float = 0.0


def _chunkings(options: EvalOptions) -> list[ChunkingStrategyName]:
    """Chunkings to index, shipped first (it warms the cache for the shipped-defaults run)."""
    wanted = {v.chunking for v in options.variants}
    first = [options.shipped.chunking] if options.shipped.chunking in wanted else []
    return first + sorted(wanted - set(first))


async def _index_all(
    services: EvalServices, dataset: BeirDataset, tenant: TenantId, options: EvalOptions
) -> tuple[
    dict[ChunkingStrategyName, IndexedCollection], dict[ChunkingStrategyName, IndexedCorpus]
]:
    targets: dict[ChunkingStrategyName, IndexedCollection] = {}
    corpora: dict[ChunkingStrategyName, IndexedCorpus] = {}
    docs = list(dataset.corpus.values())
    for chunking in _chunkings(options):
        patch = SettingsPatch(chunking={"strategy": chunking}, language=Language.ENGLISH)
        change = await services.collections.create(tenant, f"{dataset.name}-{chunking}", patch)
        cid = change.collection.record.id
        options.log(f"indexing {len(docs)} documents ({chunking} chunking)")
        step = max(1, len(docs) // 10)

        def progress(done: int, total: int, label: str = str(chunking), every: int = step) -> None:
            if done % every == 0 or done == total:
                options.log(f"  {label}: {done}/{total}")

        corpus = await index_corpus(
            services.uploader,
            services.ingest,
            tenant,
            cid,
            docs,
            concurrency=options.concurrency,
            progress=progress,
        )
        options.log(
            f"indexed {chunking}: {corpus.chunks} chunks, {corpus.embedded} texts embedded, "
            f"{corpus.seconds:.0f}s"
        )
        if options.on_indexed is not None:
            options.on_indexed(chunking, corpus)
        corpora[chunking] = corpus
        targets[chunking] = IndexedCollection(tenant, cid, corpus.doc_ids)
    return targets, corpora


async def _baselines(
    dataset: BeirDataset,
    options: EvalOptions,
    vector_runs: Mapping[str, RunOutput],
    splits: Sequence[str],
) -> list[Row]:
    """`bm25s` per split, and `bm25s` + vector RRF where the shipped vector row ran."""
    if not options.bm25:
        return []
    options.log("bm25s baseline")
    index = Bm25Index(dataset.corpus)
    retrieval = options.shipped.retrieval
    rows: list[Row] = []
    for split in splits:
        queries = dict(options.queries(dataset, split))
        bm25 = index.run(queries, EVAL_DEPTH)
        qrels = dataset.qrels[split]
        output = RunOutput(bm25.rankings, bm25.seconds)
        rows.append(
            make_row(
                "bm25s",
                split=split,
                mode="bm25s",
                chunking="none",
                settings="bm25s lucene k1=1.5 b=0.75, English stopwords, Snowball stemmer",
                output=output,
                qrels=qrels,
                group="baselines",
                note="reference, in-process",
            )
        )
        vector = vector_runs.get(split)
        if vector is None:
            continue
        weights = retrieval.rrf_weights
        fused = {
            q: fuse_rankings(
                [
                    (bm25.rankings[q], float(weights.get("lexical", 0.0))),
                    (vector.rankings[q], float(weights.get("vector", 1.0))),
                ],
                retrieval.rrf_k,
                tiebreak=vector.rankings[q],
            )[:EVAL_DEPTH]
            for q in vector.rankings
        }
        seconds = {q: bm25.seconds[q] + vector.seconds[q] for q in vector.rankings}
        rows.append(
            make_row(
                "bm25s + vector RRF",
                split=split,
                mode="bm25s+vector",
                chunking=str(options.shipped.chunking),
                settings=(
                    f"rrf_k={retrieval.rrf_k} rrf_weights=lexical:{weights.get('lexical', 0.0):g}"
                    f",vector:{weights.get('vector', 1.0):g}; document-level fusion"
                ),
                output=RunOutput(fused, seconds),
                qrels=qrels,
                group="baselines",
                note="reference, not shippable",
            )
        )
    return rows


async def run_retrieval_eval(
    services: EvalServices, dataset: BeirDataset, options: EvalOptions
) -> EvalOutcome:
    """Index, run every variant, add the baselines; delete the tenant unless `keep`."""
    started = time.perf_counter()
    splits = list(dict.fromkeys(v.split for v in options.variants))
    queries = {s: options.queries(dataset, s) for s in splits}
    for v in options.variants:
        if v.mode is EvalMode.HYBRID_RERANK and v.reranker_variant not in services.rerankers:
            raise EvalError(f"{v.name} needs the {v.reranker_variant} reranker (rerank.provider)")
    tenant = (await services.admin.create_tenant(f"eval-{dataset.name}-{options.run_id}")).id
    options.log(f"temporary tenant {tenant}")
    outcome = EvalOutcome([], tenant, options.keep)
    try:
        targets, outcome.corpora = await _index_all(services, dataset, tenant, options)
        vector_runs: dict[str, RunOutput] = {}
        for variant in options.variants:
            options.log(f"running {variant.name} on {variant.split}")
            output = await run_variant(
                services.search,
                targets[variant.chunking],
                queries[variant.split],
                variant,
                services.rerankers,
            )
            shipped = variant.is_shipped(options.shipped)
            if variant.mode is EvalMode.VECTOR and variant.chunking == options.shipped.chunking:
                vector_runs[variant.split] = output
            outcome.rows.append(
                make_row(
                    variant.name,
                    split=variant.split,
                    mode=variant.mode.value,
                    chunking=str(variant.chunking),
                    settings=variant.settings(options.shipped.retrieval),
                    output=output,
                    qrels=dataset.qrels[variant.split],
                    shipped=shipped,
                    group=variant.group,
                )
            )
        outcome.rows.extend(await _baselines(dataset, options, vector_runs, splits))
    finally:
        if not options.keep:
            options.log(f"deleting temporary tenant {tenant}")
            await services.delete_tenant(tenant)
    outcome.seconds = time.perf_counter() - started
    return outcome
