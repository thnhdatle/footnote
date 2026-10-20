# SPDX-License-Identifier: Apache-2.0
"""Retrieval values: hits, ranked legs, weighted RRF, lexical query plan, pooling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from footnote.domain.documents import ChunkId, DocumentId
from footnote.domain.settings import Language, LexicalMatch, LexicalRank

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_ANY_TEMPLATE = (
    "(SELECT to_tsquery('simple', string_agg(quote_literal(lex), ' | ' ORDER BY lex)) "
    "FROM unnest(tsvector_to_array(to_tsvector({lang}, %(query)s))) AS lex)"
)
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class LegHit:
    """One row of a retrieval leg; `score` is ts_rank or cosine similarity."""

    chunk_id: ChunkId
    score: float


@dataclass(frozen=True, slots=True)
class RankedList:
    """Output of one leg (`lexical` or `vector`), best first."""

    leg: str
    hits: tuple[LegHit, ...]


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A chunk after fusion; `ranks` and `scores` are 1-based per leg where present."""

    chunk_id: ChunkId
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """A retrieved chunk with display data and per-stage scores."""

    chunk_id: ChunkId
    document_id: DocumentId
    title: str
    heading_path: tuple[str, ...]
    text: str
    embedded_text: str
    page_from: int | None
    page_to: int | None
    rrf_score: float | None = None
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


def _default_weights() -> dict[str, float]:
    return {"lexical": 0.2, "vector": 1.0}


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Weighted reciprocal-rank-fusion parameters; `weights` is keyed by leg name."""

    rrf_k: int = 60
    weights: Mapping[str, float] = field(default_factory=_default_weights)


def rrf(lists: Sequence[RankedList], config: FusionConfig) -> list[FusedHit]:
    """Fuse ranked legs by weighted RRF.

    Ties break on vector score (desc, chunks absent from the vector leg last), then chunk id.
    """
    totals: dict[ChunkId, float] = {}
    ranks: dict[ChunkId, dict[str, int]] = {}
    scores: dict[ChunkId, dict[str, float]] = {}
    for ranked in lists:
        if ranked.leg not in config.weights:
            msg = f"no fusion weight configured for leg {ranked.leg!r}"
            raise ValueError(msg)
        weight = config.weights[ranked.leg]
        for position, hit in enumerate(ranked.hits, start=1):
            leg_ranks = ranks.setdefault(hit.chunk_id, {})
            if ranked.leg in leg_ranks:
                continue
            leg_ranks[ranked.leg] = position
            scores.setdefault(hit.chunk_id, {})[ranked.leg] = hit.score
            totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + weight / (
                config.rrf_k + position
            )

    def sort_key(chunk_id: ChunkId) -> tuple[float, bool, float, str]:
        vector = scores[chunk_id].get("vector")
        return (
            -totals[chunk_id],
            vector is None,
            0.0 if vector is None else -vector,
            str(chunk_id),
        )

    return [
        FusedHit(chunk_id, totals[chunk_id], ranks[chunk_id], scores[chunk_id])
        for chunk_id in sorted(totals, key=sort_key)
    ]


def lexical_tsquery_sql(match: LexicalMatch, language: Language) -> str:
    """SQL tsquery expression for the lexical leg; the query text is `%(query)s`."""
    lang = f"'{language.value}'::regconfig"
    if match is LexicalMatch.ANY:
        return _ANY_TEMPLATE.replace("{lang}", lang)
    if match is LexicalMatch.ALL:
        return f"plainto_tsquery({lang}, %(query)s)"
    return f"websearch_to_tsquery({lang}, %(query)s)"


def lexical_rank_sql(rank: LexicalRank, tsquery_alias: str = "q") -> str:
    """SQL rank expression over `c.tsv` and the tsquery alias."""
    if not _IDENTIFIER.fullmatch(tsquery_alias):
        msg = f"invalid tsquery alias: {tsquery_alias!r}"
        raise ValueError(msg)
    return f"{rank.value}(c.tsv, {tsquery_alias})"


def pool_documents[T](document_ids: Iterable[T]) -> list[T]:
    """Distinct ids in first-occurrence order (max pooling of a ranked list)."""
    return list(dict.fromkeys(document_ids))


def eval_order[T](
    fused: Sequence[T],
    reranked: Sequence[T] | None,
    rerank_n: int,
) -> list[T]:
    """Final evaluation order: reranked head (up to `rerank_n`), then the fused remainder."""
    if reranked is None:
        return list(fused)
    head = list(reranked[:rerank_n])
    seen = set(head)
    return head + [item for item in fused if item not in seen]


def select_vector_path(chunk_count: int, exact_scan_max: int) -> str:
    """`exact` for small corpora, else `hnsw`."""
    return "exact" if chunk_count <= exact_scan_max else "hnsw"


def ef_search(configured: int, k_vector: int) -> int:
    """HNSW `ef_search` must be at least the number of vector candidates requested."""
    return max(configured, k_vector)
