# SPDX-License-Identifier: Apache-2.0
"""Retrieval values: hits, ranked legs, weighted RRF and the lexical query plan."""

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
    chunk_id: ChunkId
    score: float


@dataclass(frozen=True, slots=True)
class RankedList:
    leg: str
    hits: tuple[LegHit, ...]


@dataclass(frozen=True, slots=True)
class FusedHit:
    chunk_id: ChunkId
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
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
    rrf_k: int = 60
    weights: Mapping[str, float] = field(default_factory=_default_weights)


def rrf(lists: Sequence[RankedList], config: FusionConfig) -> list[FusedHit]:
    """Fuses ranked legs by weighted reciprocal rank fusion."""
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
    lang = f"'{language.value}'::regconfig"
    if match is LexicalMatch.ANY:
        return _ANY_TEMPLATE.replace("{lang}", lang)
    if match is LexicalMatch.ALL:
        return f"plainto_tsquery({lang}, %(query)s)"
    return f"websearch_to_tsquery({lang}, %(query)s)"


def lexical_rank_sql(rank: LexicalRank, tsquery_alias: str = "q") -> str:
    if not _IDENTIFIER.fullmatch(tsquery_alias):
        msg = f"invalid tsquery alias: {tsquery_alias!r}"
        raise ValueError(msg)
    return f"{rank.value}(c.tsv, {tsquery_alias})"


def pool_documents[T](document_ids: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(document_ids))


def eval_order[T](
    fused: Sequence[T],
    reranked: Sequence[T] | None,
    rerank_n: int,
) -> list[T]:
    if reranked is None:
        return list(fused)
    head = list(reranked[:rerank_n])
    seen = set(head)
    return head + [item for item in fused if item not in seen]


def select_vector_path(chunk_count: int, exact_scan_max: int) -> str:
    return "exact" if chunk_count <= exact_scan_max else "hnsw"


def ef_search(configured: int, k_vector: int) -> int:
    return max(configured, k_vector)
