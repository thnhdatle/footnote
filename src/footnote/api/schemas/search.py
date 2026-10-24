# SPDX-License-Identifier: Apache-2.0
"""Schemas of the search and answer routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from footnote.api.schemas.common import Request
from footnote.app.answer import QUESTION_MAX_CHARS
from footnote.app.search import SearchResult
from footnote.domain.answer import AbstentionReason, Answer, Citation
from footnote.domain.documents import DocumentId
from footnote.domain.ports import SearchFilters
from footnote.domain.retrieval import Hit
from footnote.domain.settings import SearchMode

QUERY_MAX_CHARS = 4000
MAX_FILTER_VALUES = 100
MAX_DOCUMENT_IDS = 1000

type Scalar = str | int | float | bool | None


class InFilter(Request):
    """Matches when the metadata value equals any listed value."""

    in_: Annotated[list[Scalar], Field(alias="in", min_length=1, max_length=MAX_FILTER_VALUES)]


class FiltersIn(Request):
    """Metadata equality / `{"in": […]}`, document ids and creation-time bounds."""

    metadata: dict[str, Scalar | InFilter] = Field(default_factory=dict, max_length=32)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_DOCUMENT_IDS)
    created_after: datetime | None = None
    created_before: datetime | None = None

    def to_domain(self) -> SearchFilters:
        metadata = {
            key: {"in": value.in_} if isinstance(value, InFilter) else value
            for key, value in self.metadata.items()
        }
        return SearchFilters(
            metadata=metadata,
            document_ids=tuple(DocumentId(d) for d in self.document_ids),
            created_after=None if self.created_after is None else self.created_after.isoformat(),
            created_before=None if self.created_before is None else self.created_before.isoformat(),
        )


def to_filters(filters: FiltersIn | None) -> SearchFilters:
    return SearchFilters() if filters is None else filters.to_domain()


class SearchIn(Request):
    """`POST /v1/collections/{id}/search`."""

    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS)
    filters: FiltersIn | None = None
    mode: SearchMode | None = Field(default=None, description="Default `retrieval.default_mode`")
    rerank: bool | None = None
    k: int | None = Field(default=None, description="1-100; default the collection's k")
    debug: bool = False


class HitDebug(BaseModel):
    """Per-stage scores and ranks (`debug=true`)."""

    rrf_score: float | None
    rerank_score: float | None
    ranks: dict[str, int]
    scores: dict[str, float]


class HitOut(BaseModel):
    """A retrieved chunk (first current-version membership's page range)."""

    chunk_id: UUID
    document_id: UUID
    title: str
    heading_path: list[str]
    page_from: int | None
    page_to: int | None
    text: str
    debug: HitDebug | None = None

    @classmethod
    def of(cls, h: Hit, *, debug: bool = False) -> HitOut:
        return cls(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            title=h.title,
            heading_path=list(h.heading_path),
            page_from=h.page_from,
            page_to=h.page_to,
            text=h.text,
            debug=HitDebug(
                rrf_score=h.rrf_score,
                rerank_score=h.rerank_score,
                ranks=dict(h.ranks),
                scores=dict(h.scores),
            )
            if debug
            else None,
        )


class SearchDebug(BaseModel):
    """Search diagnostics (`debug=true`)."""

    space: str
    reranked: bool
    exact_scan_reason: str | None
    stage_seconds: dict[str, float]


class SearchOut(BaseModel):
    """Search response."""

    hits: list[HitOut]
    debug: SearchDebug | None = None

    @classmethod
    def of(cls, r: SearchResult, *, debug: bool) -> SearchOut:
        return cls(
            hits=[HitOut.of(h, debug=debug) for h in r.hits],
            debug=SearchDebug(
                space=r.space.name,
                reranked=r.reranked,
                exact_scan_reason=r.exact_scan_reason,
                stage_seconds=dict(r.stage_seconds),
            )
            if debug
            else None,
        )


class AnswerIn(Request):
    """`POST /v1/collections/{id}/answers`; `stream: true` answers with SSE events."""

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    filters: FiltersIn | None = None
    mode: SearchMode | None = None
    rerank: bool | None = None
    stream: bool = False


class CitationOut(BaseModel):
    """A verified citation; quote offsets index into the cited chunk's text."""

    n: int
    chunk_id: UUID
    document_id: UUID
    title: str
    page_from: int | None
    page_to: int | None
    quote_start: int
    quote_end: int

    @classmethod
    def of(cls, c: Citation) -> CitationOut:
        return cls(
            n=c.n,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            title=c.title,
            page_from=c.page_from,
            page_to=c.page_to,
            quote_start=c.quote_start,
            quote_end=c.quote_end,
        )


class UsageOut(BaseModel):
    """LLM token usage."""

    input_tokens: int
    output_tokens: int


class AnswerOut(BaseModel):
    """Answer with verified citations, or an abstention (`answer: null`)."""

    answer: str | None
    citations: list[CitationOut]
    abstained: bool
    abstention_reason: AbstentionReason | None
    unsupported: bool
    evidence_score: float | None
    usage: UsageOut
    trace_id: str | None = None

    @classmethod
    def of(cls, a: Answer, trace_id: str | None) -> AnswerOut:
        return cls(
            answer=a.text,
            citations=[CitationOut.of(c) for c in a.citations],
            abstained=a.abstained,
            abstention_reason=a.abstention_reason,
            unsupported=a.unsupported,
            evidence_score=a.gate_score,
            usage=UsageOut(input_tokens=a.usage.input_tokens, output_tokens=a.usage.output_tokens),
            trace_id=trace_id,
        )
