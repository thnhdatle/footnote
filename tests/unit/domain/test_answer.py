# SPDX-License-Identifier: Apache-2.0
"""Unit tests for answer value objects."""

from __future__ import annotations

from uuid import UUID

from footnote.domain.answer import AbstentionReason, Answer, Citation, Usage
from footnote.domain.documents import ChunkId, DocumentId


def test_answer_defaults_are_not_abstained() -> None:
    answer = Answer(text="x")
    assert answer.abstained is False
    assert answer.abstention_reason is None
    assert answer.citations == ()
    assert answer.usage == Usage(0, 0)
    assert answer.gate_score is None


def test_abstention_reason_values() -> None:
    assert AbstentionReason.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
    assert AbstentionReason.UNSUPPORTED.value == "unsupported"


def test_citation_holds_quote_offsets() -> None:
    citation = Citation(1, ChunkId(UUID(int=1)), DocumentId(UUID(int=2)), "T", 1, 2, 3, 9)
    assert (citation.quote_start, citation.quote_end) == (3, 9)
