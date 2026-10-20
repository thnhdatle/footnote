# SPDX-License-Identifier: Apache-2.0
"""Answers, citations and abstention reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from footnote.domain.documents import ChunkId, DocumentId


class AbstentionReason(StrEnum):
    """Why no answer was generated."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified `[n]` marker; quote offsets index into the cited chunk's text."""

    n: int
    chunk_id: ChunkId
    document_id: DocumentId
    title: str
    page_from: int | None
    page_to: int | None
    quote_start: int
    quote_end: int


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage reported by the LLM."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Answer:
    """Result of `/answers`: `text` is None when abstained."""

    text: str | None
    citations: tuple[Citation, ...] = ()
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    unsupported: bool = False
    usage: Usage = field(default_factory=Usage)
    gate_score: float | None = None
