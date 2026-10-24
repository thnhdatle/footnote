# SPDX-License-Identifier: Apache-2.0
"""Evidence gate scoring, threshold resolution and calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from footnote.domain.errors import ConflictError

if TYPE_CHECKING:
    from collections.abc import Sequence

_EPSILON = 1e-9


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class GateKind(StrEnum):
    RERANK = "rerank"
    VECTOR = "vector"


def rerank_gate_score(logits: Sequence[float]) -> float | None:
    if not logits:
        return None
    return sigmoid(max(logits))


def vector_gate_score(similarities: Sequence[float]) -> float | None:
    if not similarities:
        return None
    return max(similarities)


def resolve_threshold(
    collection: float | None,
    config: float | None,
    calibrated: float | None,
    kind: GateKind,
) -> float:
    for value in (collection, config, calibrated):
        if value is not None:
            return value
    raise ConflictError(
        f"no {kind.value} evidence threshold is set or calibrated",
        code="evidence_threshold_uncalibrated",
    )


@dataclass(frozen=True)
class GateDecision:
    kind: GateKind
    score: float | None
    threshold: float
    passed: bool


def decide(kind: GateKind, score: float | None, threshold: float) -> GateDecision:
    """Passes when a score exists and reaches the threshold."""
    return GateDecision(
        kind=kind,
        score=score,
        threshold=threshold,
        passed=score is not None and score >= threshold,
    )


def calibrate_threshold(scores: Sequence[float], target_answer_rate: float) -> float:
    """Returns the largest threshold that reaches the target answer rate."""
    if not scores:
        raise ValueError("scores must not be empty")
    if not 0.0 < target_answer_rate <= 1.0:
        raise ValueError("target_answer_rate must be in (0, 1]")
    ordered = sorted(scores, reverse=True)
    index = max(math.ceil(target_answer_rate * len(ordered) - _EPSILON) - 1, 0)
    return ordered[index]
