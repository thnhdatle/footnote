# SPDX-License-Identifier: Apache-2.0
"""Retrieval and abstention metrics (pure, stdlib only; pytrec_eval-compatible definitions)."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_CURVE_STEPS = 100
_PERCENT = 100.0


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Share of relevant documents (grade > 0) found in the top k; 0.0 without any."""
    relevant = sum(1 for rel in qrels.values() if rel > 0)
    if relevant == 0:
        return 0.0
    found = sum(1 for doc in ranked[:k] if qrels.get(doc, 0) > 0)
    return found / relevant


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Normalised DCG at k with linear gain (pytrec_eval `ndcg_cut`); 0.0 when IDCG is 0."""
    ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)
    idcg = _dcg(ideal[:k])
    if idcg == 0.0:
        return 0.0
    gains = [max(qrels.get(doc, 0), 0) for doc in ranked[:k]]
    return _dcg(gains) / idcg


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top k, else 0.0."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if qrels.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; ValueError on empty input."""
    if not values:
        raise ValueError("values must not be empty")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile `p` in [0, 100] with linear interpolation (numpy `method="linear"`)."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= p <= _PERCENT:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    position = p / _PERCENT * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(positive > negative) + 0.5 P(tie), computed in O(n log n)."""
    if not positive or not negative:
        raise ValueError("positive and negative scores must not be empty")
    ordered = sorted(negative)
    wins = 0
    ties = 0
    for score in positive:
        low = bisect_left(ordered, score)
        wins += low
        ties += bisect_right(ordered, score) - low
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


@dataclass(frozen=True)
class CurvePoint:
    """Answer and abstain rates at one gate threshold."""

    threshold: float
    answer_rate: float
    abstain_rate: float


def rates_at(
    answerable: Sequence[float], unanswerable: Sequence[float], threshold: float
) -> CurvePoint:
    """Answer rate on answerable scores and abstain rate on unanswerable scores."""
    answered = sum(1 for s in answerable if s >= threshold)
    abstained = sum(1 for s in unanswerable if s < threshold)
    return CurvePoint(
        threshold=threshold,
        answer_rate=answered / len(answerable) if answerable else 0.0,
        abstain_rate=abstained / len(unanswerable) if unanswerable else 0.0,
    )


def abstention_curve(
    answerable: Sequence[float], unanswerable: Sequence[float]
) -> list[CurvePoint]:
    """Rates at thresholds 0.00, 0.01, ..., 1.00 (101 points)."""
    return [rates_at(answerable, unanswerable, i / _CURVE_STEPS) for i in range(_CURVE_STEPS + 1)]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Mean retrieval metrics over a query set."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    mrr_at_10: float

    def as_dict(self) -> dict[str, float]:
        """Metric name to value."""
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "ndcg_at_10": self.ndcg_at_10,
            "mrr_at_10": self.mrr_at_10,
        }


def evaluate_run(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, float]:
    """Average metrics over the qids in `qrels`; a qid missing from `run` scores 0."""
    if not qrels:
        raise ValueError("qrels must not be empty")
    rankings = {qid: run.get(qid, []) for qid in qrels}

    def avg(metric: str, k: int) -> float:
        fn = {"recall": recall_at_k, "ndcg": ndcg_at_k, "mrr": mrr_at_k}[metric]
        return mean([fn(rankings[qid], judged, k) for qid, judged in qrels.items()])

    return RetrievalMetrics(
        recall_at_1=avg("recall", 1),
        recall_at_5=avg("recall", 5),
        recall_at_10=avg("recall", 10),
        recall_at_100=avg("recall", 100),
        ndcg_at_10=avg("ndcg", 10),
        mrr_at_10=avg("mrr", 10),
    ).as_dict()
