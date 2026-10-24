# SPDX-License-Identifier: Apache-2.0
"""Search use case: embed the query, run the legs, fuse and rerank."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from footnote.domain.documents import ChunkId, CollectionId, DocumentId
from footnote.domain.embedding import EmbeddingSpace, l2_normalise
from footnote.domain.errors import ValidationError
from footnote.domain.ports import (
    Clock,
    CollectionView,
    EmbedderFactory,
    LexicalQuery,
    Reranker,
    SearchFilters,
    SearchSnapshot,
    SearchStore,
    VectorQuery,
)
from footnote.domain.retrieval import (
    FusedHit,
    FusionConfig,
    Hit,
    LegHit,
    RankedList,
    eval_order,
    rrf,
    select_vector_path,
)
from footnote.domain.settings import K_MAX, LexicalMatch, LexicalRank, RetrievalSettings, SearchMode
from footnote.domain.tenancy import TenantId

LEXICAL = "lexical"
VECTOR = "vector"


@dataclass(frozen=True, slots=True)
class SearchOverrides:
    lexical_match: LexicalMatch | None = None
    lexical_rank: LexicalRank | None = None
    rrf_weights: Mapping[str, float] | None = None
    rerank_n: int | None = None
    reranker: Reranker | None = None

    def apply(self, retrieval: RetrievalSettings) -> RetrievalSettings:
        return replace(
            retrieval,
            lexical_match=self.lexical_match or retrieval.lexical_match,
            lexical_rank=self.lexical_rank or retrieval.lexical_rank,
            rrf_weights=(
                retrieval.rrf_weights
                if self.rrf_weights is None
                else {**retrieval.rrf_weights, **self.rrf_weights}
            ),
            rerank_n=retrieval.rerank_n if self.rerank_n is None else self.rerank_n,
        )


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    mode: SearchMode | None = None
    rerank: bool | None = None
    k: int | None = None
    filters: SearchFilters = field(default_factory=SearchFilters)
    need_similarity: bool = False
    depth: int | None = None
    leg_limit: int | None = None
    overrides: SearchOverrides | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[Hit]
    collection: CollectionView
    space: EmbeddingSpace
    reranked: bool
    rerank_logits: list[float]
    similarities: dict[ChunkId, float]
    ranking: list[ChunkId]
    exact_scan_reason: str | None
    stage_seconds: dict[str, float]
    ranked_documents: list[DocumentId] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    exact_scan_max: int = 20_000
    default_mode: SearchMode = SearchMode.HYBRID


@dataclass(frozen=True, slots=True)
class _Plan:
    request: SearchRequest
    mode: SearchMode
    use_rerank: bool
    timings: dict[str, float]
    reranker: Reranker | None = None


@dataclass(slots=True)
class _Legs:
    lists: list[RankedList]
    exact_scan_reason: str | None
    vector: list[float] | None


class SearchService:
    """Runs searches against a search store."""

    def __init__(
        self,
        store: SearchStore,
        embedders: EmbedderFactory,
        reranker: Reranker | None,
        clock: Clock,
        config: SearchConfig | None = None,
    ) -> None:
        self._store = store
        self._embedders = embedders
        self._reranker = reranker
        self._clock = clock
        self._config = config or SearchConfig()

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    async def search(
        self, tenant: TenantId, collection_id: CollectionId, request: SearchRequest
    ) -> SearchResult:
        """Runs one search in a read snapshot and reranks the result."""
        if not request.query.strip():
            raise ValidationError("query", "must not be empty")
        timings: dict[str, float] = {}
        overrides = request.overrides
        reranker = self._reranker
        if overrides is not None and overrides.reranker is not None:
            reranker = overrides.reranker
        async with self._store.snapshot(tenant, collection_id) as snap:
            view = await snap.collection()
            if overrides is not None:
                retrieval = overrides.apply(view.settings.retrieval)
                view = replace(view, settings=replace(view.settings, retrieval=retrieval))
            retrieval = view.settings.retrieval
            k = request.k if request.k is not None else retrieval.k
            if not 1 <= k <= K_MAX:
                raise ValidationError("k", f"must be in [1, {K_MAX}]")
            mode = request.mode or self._config.default_mode
            use_rerank = (retrieval.rerank if request.rerank is None else request.rerank) and (
                reranker is not None
            )
            plan = _Plan(request, mode, use_rerank, timings, reranker)
            legs = await self._run_legs(snap, view, plan)
            fused = _fuse(legs.lists, view, mode)
            depth = max(k, retrieval.rerank_n if use_rerank else 0, request.depth or 0)
            candidates = [f.chunk_id for f in fused[:depth]]
            hits = _annotate(await snap.hits(candidates), fused)
            sims: dict[ChunkId, float] = {}
            if request.need_similarity and legs.vector is not None and candidates:
                sims = await snap.similarities(candidates, legs.vector)
        fused_ids = [h.chunk_id for h in hits]
        logits: list[float] = []
        reranked_ids: list[ChunkId] | None = None
        if use_rerank and hits:
            start = self._clock.monotonic()
            hits, logits = await self._rerank(plan, hits, retrieval.rerank_n)
            reranked_ids = [h.chunk_id for h in hits[: retrieval.rerank_n]]
            timings["rerank"] = self._clock.monotonic() - start
        ranking = eval_order(fused_ids, reranked_ids, retrieval.rerank_n)
        by_id = {h.chunk_id: h for h in hits}
        if reranked_ids is not None:
            hits = [by_id[c] for c in ranking]
        ranking = ranking[: max(depth, k)]
        return SearchResult(
            hits=hits[:k],
            collection=view,
            space=view.active_space,
            reranked=use_rerank,
            rerank_logits=logits,
            similarities=sims,
            ranking=ranking,
            exact_scan_reason=legs.exact_scan_reason,
            stage_seconds=timings,
            ranked_documents=[by_id[c].document_id for c in ranking],
        )

    async def _run_legs(self, snap: SearchSnapshot, view: CollectionView, plan: _Plan) -> _Legs:
        request, mode, timings = plan.request, plan.mode, plan.timings
        retrieval = view.settings.retrieval
        lists: list[RankedList] = []
        reason: str | None = None
        vector: list[float] | None = None
        need_vector = mode in (SearchMode.VECTOR, SearchMode.HYBRID) or (
            request.need_similarity and not plan.use_rerank
        )
        if need_vector:
            start = self._clock.monotonic()
            raw = await self._embedders.for_space(view.active_space).embed_query(request.query)
            vector = l2_normalise(raw)
            timings["embed_query"] = self._clock.monotonic() - start
        if mode in (SearchMode.LEXICAL, SearchMode.HYBRID):
            start = self._clock.monotonic()
            limit = max(retrieval.k_lexical, request.leg_limit or 0)
            lexical = await snap.lexical(
                LexicalQuery(request.query, retrieval.lexical_match, retrieval.lexical_rank, limit),
                request.filters,
            )
            lists.append(RankedList(LEXICAL, tuple(lexical)))
            timings["lexical"] = self._clock.monotonic() - start
        if vector is not None and mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            start = self._clock.monotonic()
            hits, reason = await self._vector_leg(snap, view, vector, request)
            lists.append(RankedList(VECTOR, tuple(hits)))
            timings["vector"] = self._clock.monotonic() - start
        return _Legs(lists, reason, vector)

    async def _vector_leg(
        self,
        snap: SearchSnapshot,
        view: CollectionView,
        vector: list[float],
        request: SearchRequest,
    ) -> tuple[list[LegHit], str | None]:
        limit = max(view.settings.retrieval.k_vector, request.leg_limit or 0)
        path = select_vector_path(view.chunk_count, self._config.exact_scan_max)
        exact = path == "exact"
        hits = await snap.vector(VectorQuery(vector, limit, exact), request.filters)
        if exact:
            return hits, "small_collection"
        if len(hits) < limit:
            return await snap.vector(VectorQuery(vector, limit, True), request.filters), "fallback"
        return hits, None

    async def _rerank(
        self, plan: _Plan, hits: list[Hit], rerank_n: int
    ) -> tuple[list[Hit], list[float]]:
        reranker = plan.reranker
        assert reranker is not None  # noqa: S101 - guarded by the caller
        head = hits[:rerank_n]
        logits = await reranker.score(plan.request.query, [h.embedded_text for h in head])
        if len(logits) != len(head):
            msg = f"reranker returned {len(logits)} scores for {len(head)} passages"
            raise RuntimeError(msg)
        scored = [replace(h, rerank_score=s) for h, s in zip(head, logits, strict=True)]
        order = sorted(range(len(scored)), key=lambda i: (-logits[i], i))
        return [scored[i] for i in order] + hits[rerank_n:], logits


def _fuse(lists: Sequence[RankedList], view: CollectionView, mode: SearchMode) -> list[FusedHit]:
    retrieval = view.settings.retrieval
    if mode is SearchMode.HYBRID:
        return rrf(lists, FusionConfig(rrf_k=retrieval.rrf_k, weights=retrieval.rrf_weights))
    return rrf(lists, FusionConfig(rrf_k=retrieval.rrf_k, weights={lst.leg: 1.0 for lst in lists}))


def _annotate(hits: Sequence[Hit], fused: Sequence[FusedHit]) -> list[Hit]:
    by_id = {f.chunk_id: f for f in fused}
    out: list[Hit] = []
    for hit in hits:
        f = by_id[hit.chunk_id]
        out.append(replace(hit, rrf_score=f.rrf_score, ranks=dict(f.ranks), scores=dict(f.scores)))
    return out
