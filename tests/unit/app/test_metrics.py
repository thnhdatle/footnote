# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footnote.app.evaluation.metrics import (
    CurvePoint,
    abstention_curve,
    evaluate_run,
    mean,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    rates_at,
    recall_at_k,
    roc_auc,
)

QRELS = {"a": 0, "b": 3, "c": 2, "e": 1}
RANKED = ["a", "b", "c", "d"]
LOG3 = math.log2(3)


def test_recall_at_k_counts_relevant_in_top_k() -> None:
    assert recall_at_k(RANKED, QRELS, 1) == 0.0
    assert recall_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 3, abs=1e-15)
    assert recall_at_k(RANKED, QRELS, 3) == pytest.approx(2 / 3, abs=1e-15)
    assert recall_at_k(RANKED, QRELS, 100) == pytest.approx(2 / 3, abs=1e-15)


def test_recall_at_k_without_relevant_is_zero() -> None:
    assert recall_at_k(RANKED, {"a": 0}, 10) == 0.0
    assert recall_at_k(RANKED, {}, 10) == 0.0


def test_ndcg_at_k_hand_example() -> None:
    expected = (3 / LOG3 + 1.0) / (3.0 + 2 / LOG3 + 0.5)
    assert ndcg_at_k(RANKED, QRELS, 3) == pytest.approx(expected, abs=1e-12)
    assert ndcg_at_k(RANKED, QRELS, 3) == pytest.approx(0.6074, abs=1e-4)


def test_ndcg_at_k_cutoff_limits_ideal_and_actual() -> None:
    assert ndcg_at_k(RANKED, QRELS, 2) == pytest.approx((3 / LOG3) / (3 + 2 / LOG3), abs=1e-12)
    assert ndcg_at_k(RANKED, QRELS, 1) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["b", "c", "e"], QRELS, 10) == pytest.approx(1.0, abs=1e-15)


def test_ndcg_at_k_no_relevant_is_zero() -> None:
    assert ndcg_at_k(RANKED, {"a": 0}, 10) == 0.0


def test_ndcg_at_k_ignores_negative_grades() -> None:
    assert ndcg_at_k(["x", "b"], {"x": -2, "b": 1}, 2) == pytest.approx(1 / LOG3, abs=1e-12)


def test_mrr_at_k_first_relevant_rank() -> None:
    assert mrr_at_k(RANKED, QRELS, 10) == 0.5
    assert mrr_at_k(["c", "b"], QRELS, 10) == 1.0


def test_mrr_at_k_respects_cutoff() -> None:
    assert mrr_at_k(RANKED, QRELS, 1) == 0.0
    assert mrr_at_k(["x", "y", "b"], QRELS, 3) == pytest.approx(1 / 3, abs=1e-15)


def test_mean_hand_example() -> None:
    assert mean([1.0, 2.0, 6.0]) == 3.0


def test_mean_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_percentile_linear_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 25) == 1.75
    assert percentile(values, 90) == pytest.approx(3.7, abs=1e-12)


def test_percentile_single_value() -> None:
    assert percentile([7.0], 99) == 7.0


def test_percentile_rejects_empty_and_bad_p() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="p must"):
        percentile([1.0], -1)
    with pytest.raises(ValueError, match="p must"):
        percentile([1.0], 100.5)


def test_roc_auc_hand_examples() -> None:
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([3.0, 5.0, 4.0], [1.0, 4.0]) == pytest.approx(4.5 / 6, abs=1e-15)


def test_roc_auc_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        roc_auc([], [1.0])
    with pytest.raises(ValueError, match="empty"):
        roc_auc([1.0], [])


@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=30),
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=30),
)
def test_roc_auc_is_antisymmetric(pos: list[float], neg: list[float]) -> None:
    assert abs(roc_auc(pos, neg) - (1 - roc_auc(neg, pos))) < 1e-12


def test_rates_at_hand_example() -> None:
    point = rates_at([0.9, 0.5, 0.2, 0.7], [0.1, 0.5, 0.6], 0.5)
    assert point == CurvePoint(threshold=0.5, answer_rate=0.75, abstain_rate=1 / 3)


def test_rates_at_empty_inputs_are_zero() -> None:
    assert rates_at([], [], 0.5) == CurvePoint(0.5, 0.0, 0.0)


def test_abstention_curve_has_101_points() -> None:
    curve = abstention_curve([0.5], [0.5])
    assert len(curve) == 101
    assert curve[0].threshold == 0.0
    assert curve[1].threshold == 0.01
    assert curve[7].threshold == 0.07
    assert curve[100].threshold == 1.0


def test_abstention_curve_values() -> None:
    curve = abstention_curve([0.9, 0.5], [0.2, 0.6])
    assert curve[0] == CurvePoint(0.0, 1.0, 0.0)
    assert curve[50] == CurvePoint(0.5, 1.0, 0.5)
    assert curve[51] == CurvePoint(0.51, 0.5, 0.5)
    assert curve[61] == CurvePoint(0.61, 0.5, 1.0)
    assert curve[91] == CurvePoint(0.91, 0.0, 1.0)


def test_evaluate_run_hand_example() -> None:
    run = {"q1": ["b", "x", "c"]}
    qrels = {"q1": {"b": 1, "c": 1}, "q2": {"z": 1}}
    result = evaluate_run(run, qrels)
    ndcg_q1 = (1 + 1 / math.log2(4)) / (1 + 1 / LOG3)
    assert result == {
        "recall_at_1": pytest.approx(0.25, abs=1e-15),
        "recall_at_5": pytest.approx(0.5, abs=1e-15),
        "recall_at_10": pytest.approx(0.5, abs=1e-15),
        "recall_at_100": pytest.approx(0.5, abs=1e-15),
        "ndcg_at_10": pytest.approx(ndcg_q1 / 2, abs=1e-12),
        "mrr_at_10": pytest.approx(0.5, abs=1e-15),
    }


def test_evaluate_run_distinguishes_cutoffs() -> None:
    ranked = [f"d{i}" for i in range(100)]
    qrels = {"q": {"d0": 1, "d4": 1, "d9": 1, "d99": 1}}
    result = evaluate_run({"q": ranked}, qrels)
    assert result["recall_at_1"] == 0.25
    assert result["recall_at_5"] == 0.5
    assert result["recall_at_10"] == 0.75
    assert result["recall_at_100"] == 1.0
    assert result["mrr_at_10"] == 1.0


def test_evaluate_run_ignores_run_qids_outside_qrels() -> None:
    result = evaluate_run({"q": ["a"], "other": ["b"]}, {"q": {"a": 1}})
    assert result["recall_at_1"] == 1.0


def test_evaluate_run_empty_qrels_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_run({}, {})
