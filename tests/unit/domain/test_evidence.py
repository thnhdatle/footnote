# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.errors import ConflictError
from footnote.domain.evidence import (
    GateDecision,
    GateKind,
    calibrate_threshold,
    decide,
    rerank_gate_score,
    resolve_threshold,
    sigmoid,
    vector_gate_score,
)


def test_sigmoid_zero_is_half() -> None:
    assert sigmoid(0.0) == 0.5


def test_sigmoid_positive_value() -> None:
    assert sigmoid(2.0) == pytest.approx(0.8807970779778823, abs=1e-15)


def test_sigmoid_negative_value() -> None:
    assert sigmoid(-2.0) == pytest.approx(0.11920292202211755, abs=1e-15)


def test_sigmoid_large_magnitudes_do_not_overflow() -> None:
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0


def test_gate_kind_values() -> None:
    assert GateKind.RERANK.value == "rerank"
    assert GateKind.VECTOR.value == "vector"


def test_rerank_gate_score_uses_max_logit() -> None:
    assert rerank_gate_score([-1.0, 2.0, 0.5]) == pytest.approx(sigmoid(2.0), abs=1e-15)


def test_rerank_gate_score_empty_is_none() -> None:
    assert rerank_gate_score([]) is None


def test_vector_gate_score_is_max() -> None:
    assert vector_gate_score([0.1, 0.7, 0.4]) == 0.7


def test_vector_gate_score_empty_is_none() -> None:
    assert vector_gate_score([]) is None


def test_resolve_threshold_prefers_collection() -> None:
    assert resolve_threshold(0.1, 0.2, 0.3, GateKind.VECTOR) == 0.1


def test_resolve_threshold_collection_zero_is_respected() -> None:
    assert resolve_threshold(0.0, 0.2, 0.3, GateKind.VECTOR) == 0.0


def test_resolve_threshold_falls_back_to_config() -> None:
    assert resolve_threshold(None, 0.2, 0.3, GateKind.VECTOR) == 0.2


def test_resolve_threshold_falls_back_to_calibrated() -> None:
    assert resolve_threshold(None, None, 0.3, GateKind.RERANK) == 0.3


def test_resolve_threshold_uncalibrated_raises_conflict() -> None:
    with pytest.raises(ConflictError) as info:
        resolve_threshold(None, None, None, GateKind.RERANK)
    assert info.value.code == "evidence_threshold_uncalibrated"
    assert info.value.message == "no rerank evidence threshold is set or calibrated"


def test_decide_passes_at_threshold() -> None:
    assert decide(GateKind.VECTOR, 0.5, 0.5) == GateDecision(GateKind.VECTOR, 0.5, 0.5, True)


def test_decide_fails_below_threshold() -> None:
    decision = decide(GateKind.RERANK, 0.49, 0.5)
    assert decision == GateDecision(GateKind.RERANK, 0.49, 0.5, False)


def test_decide_fails_without_score() -> None:
    decision = decide(GateKind.RERANK, None, -5.0)
    assert decision.passed is False
    assert decision.score is None


def test_calibrate_threshold_hand_example() -> None:
    scores = [0.9, 0.1, 0.5, 0.7, 0.3]
    assert calibrate_threshold(scores, 0.2) == 0.9
    assert calibrate_threshold(scores, 0.4) == 0.7
    assert calibrate_threshold(scores, 0.6) == 0.5
    assert calibrate_threshold(scores, 0.61) == 0.3
    assert calibrate_threshold(scores, 1.0) == 0.1


def test_calibrate_threshold_guards_float_error() -> None:
    scores = [float(i) for i in range(1, 101)]
    assert 0.07 * 100 > 7.0
    assert calibrate_threshold(scores, 0.07) == 94.0


def test_calibrate_threshold_tiny_target_clamps_to_best() -> None:
    assert calibrate_threshold([1.0, 2.0], 1e-12) == 2.0


def test_calibrate_threshold_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate_threshold([], 0.5)


@pytest.mark.parametrize("target", [0.0, -0.1, 1.01])
def test_calibrate_threshold_rejects_bad_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_answer_rate"):
        calibrate_threshold([0.5], target)


@given(
    st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=50),
    st.floats(0.01, 1.0, allow_nan=False),
)
def test_calibrate_threshold_is_largest_threshold_meeting_target(
    scores: list[float], target: float
) -> None:
    result = calibrate_threshold(scores, target)
    n = len(scores)
    assert sum(s >= result for s in scores) >= math.ceil(target * n - 1e-9)
    larger = [s for s in scores if s > result]
    if larger:
        nxt = min(larger)
        assert sum(s >= nxt for s in scores) < math.ceil(target * n - 1e-9)
