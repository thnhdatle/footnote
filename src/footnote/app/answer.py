# SPDX-License-Identifier: Apache-2.0
"""Answer use case: search → evidence gate → generate → verify citations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from footnote.app.prompts_loader import load_template
from footnote.app.search import SearchOverrides, SearchRequest, SearchResult, SearchService
from footnote.domain.answer import AbstentionReason, Answer, Usage
from footnote.domain.citations import strip_reasoning, verify
from footnote.domain.documents import CollectionId
from footnote.domain.errors import UnavailableError, ValidationError
from footnote.domain.evidence import (
    GateDecision,
    GateKind,
    decide,
    rerank_gate_score,
    resolve_threshold,
    vector_gate_score,
)
from footnote.domain.ports import LLM, Reranker, SearchFilters
from footnote.domain.settings import SearchMode
from footnote.domain.tenancy import TenantId

QUESTION_MAX_CHARS = 4000


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """One question."""

    question: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    mode: SearchMode | None = None
    rerank: bool | None = None


@dataclass(frozen=True, slots=True)
class AnswerConfig:
    """Service-level answer options (config `answer.*`)."""

    evidence_threshold_rerank: float | None = None
    evidence_threshold_vector: float | None = None


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    """The answer plus the search and gate that produced it."""

    answer: Answer
    search: SearchResult
    gate: GateDecision


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    """One SSE event: `retrieval`, `delta`, `citations`, `done` or `error`."""

    event: str
    data: dict[str, Any]


class AnswerService:
    """Answers questions with verified citations, or abstains."""

    def __init__(self, search: SearchService, llm: LLM, config: AnswerConfig | None = None) -> None:
        self._search = search
        self._llm = llm
        self._config = config or AnswerConfig()

    async def gate(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        request: AnswerRequest,
        *,
        overrides: SearchOverrides | None = None,
    ) -> tuple[SearchResult, GateDecision]:
        """Search and evaluate the evidence gate without calling the LLM.

        `overrides` is evaluation-only (`footnote eval calibrate` scores other reranker files
        through this same gate); the API never sets it.
        """
        question = request.question.strip()
        if not question or len(question) > QUESTION_MAX_CHARS:
            raise ValidationError("question", f"must be 1..{QUESTION_MAX_CHARS} characters")
        result = await self._search.search(
            tenant,
            collection_id,
            SearchRequest(
                query=question,
                mode=request.mode,
                rerank=request.rerank,
                filters=request.filters,
                need_similarity=True,
                overrides=overrides,
            ),
        )
        override = overrides.reranker if overrides is not None else None
        return result, self._decide(result, override)

    def _decide(self, result: SearchResult, override: Reranker | None = None) -> GateDecision:
        settings = result.collection.settings.answer
        if result.reranked:
            reranker = override or self._search.reranker
            calibrated = reranker.calibrated_threshold if reranker is not None else None
            threshold = resolve_threshold(
                settings.evidence_threshold_rerank,
                self._config.evidence_threshold_rerank,
                calibrated,
                GateKind.RERANK,
            )
            return decide(GateKind.RERANK, rerank_gate_score(result.rerank_logits), threshold)
        threshold = resolve_threshold(
            settings.evidence_threshold_vector,
            self._config.evidence_threshold_vector,
            result.space.calibrated_threshold,
            GateKind.VECTOR,
        )
        sims = [
            result.similarities[h.chunk_id]
            for h in result.hits
            if h.chunk_id in result.similarities
        ]
        return decide(GateKind.VECTOR, vector_gate_score(sims), threshold)

    def _require_llm(self) -> None:
        if not self._llm.configured:
            raise UnavailableError("llm.base_url is empty", code="llm_not_configured")

    async def answer(
        self, tenant: TenantId, collection_id: CollectionId, request: AnswerRequest
    ) -> AnswerOutcome:
        """Answer one question (non-streamed)."""
        self._require_llm()
        result, decision = await self.gate(tenant, collection_id, request)
        if not decision.passed:
            return AnswerOutcome(_abstain(decision), result, decision)
        template = load_template(result.collection.settings.answer.prompt_template)
        messages = template.render(request.question.strip(), result.hits)
        completion = await self._llm.chat(messages)
        answer = _finish(completion.text, result, decision, completion.usage)
        return AnswerOutcome(answer, result, decision)

    async def stream(
        self, tenant: TenantId, collection_id: CollectionId, request: AnswerRequest
    ) -> AsyncIterator[AnswerEvent]:
        """Stream `retrieval`, `delta`…, `citations`, `done` (or `error`) events."""
        self._require_llm()
        result, decision = await self.gate(tenant, collection_id, request)
        yield AnswerEvent("retrieval", {"hits": result.hits})
        if not decision.passed:
            answer = _abstain(decision)
            yield AnswerEvent("citations", {"citations": [], "answer": None, "unsupported": False})
            yield AnswerEvent("done", _done(answer))
            return
        template = load_template(result.collection.settings.answer.prompt_template)
        messages = template.render(request.question.strip(), result.hits)
        parts: list[str] = []
        usage = Usage()
        try:
            async for delta in self._llm.stream(messages):
                if delta.text:
                    parts.append(delta.text)
                    yield AnswerEvent("delta", {"text": delta.text})
                if delta.usage is not None:
                    usage = delta.usage
        except Exception as exc:  # reported to the client as an `error` event
            code = getattr(exc, "code", "upstream_failed")
            yield AnswerEvent("error", {"code": code, "message": str(exc)})
            return
        answer = _finish("".join(parts), result, decision, usage)
        yield AnswerEvent(
            "citations",
            {
                "citations": list(answer.citations),
                "answer": answer.text,
                "unsupported": answer.unsupported,
            },
        )
        yield AnswerEvent("done", _done(answer))


def _abstain(decision: GateDecision) -> Answer:
    return Answer(
        text=None,
        abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        gate_score=decision.score,
    )


def _finish(raw: str, result: SearchResult, decision: GateDecision, usage: Usage) -> Answer:
    verified = verify(strip_reasoning(raw), result.hits)
    if verified.unsupported and result.collection.settings.answer.unsupported_as_abstain:
        return Answer(
            text=None,
            abstained=True,
            abstention_reason=AbstentionReason.UNSUPPORTED,
            unsupported=True,
            usage=usage,
            gate_score=decision.score,
        )
    return Answer(
        text=verified.text,
        citations=verified.citations,
        unsupported=verified.unsupported,
        usage=usage,
        gate_score=decision.score,
    )


def _done(answer: Answer) -> dict[str, Any]:
    return {
        "usage": {
            "input_tokens": answer.usage.input_tokens,
            "output_tokens": answer.usage.output_tokens,
        },
        "abstained": answer.abstained,
        "abstention_reason": answer.abstention_reason,
        "unsupported": answer.unsupported,
    }
