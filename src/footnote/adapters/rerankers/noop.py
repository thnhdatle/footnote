# SPDX-License-Identifier: Apache-2.0
"""Reranker that keeps the fused order."""

from __future__ import annotations

from collections.abc import Sequence


class NoopReranker:
    """Reranker that keeps the input order."""

    model_key = "noop"
    calibrated_threshold: float | None = None

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:  # noqa: ARG002
        return [float(-i) for i in range(len(passages))]
