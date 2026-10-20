# SPDX-License-Identifier: Apache-2.0
"""Reranker that keeps the fused order (`rerank.provider: noop`)."""

from __future__ import annotations

from collections.abc import Sequence


class NoopReranker:
    """Implements `ports.Reranker` with strictly decreasing scores, so sorting is a no-op."""

    model_key = "noop"
    calibrated_threshold: float | None = None

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:  # noqa: ARG002
        """Scores that preserve the input order."""
        return [float(-i) for i in range(len(passages))]
