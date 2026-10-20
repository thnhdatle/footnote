# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the answer use case: gate, generation, citations and streaming."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import pytest

from footnote.app.answer import (
    QUESTION_MAX_CHARS,
    AnswerConfig,
    AnswerEvent,
    AnswerRequest,
    AnswerService,
)
from footnote.app.search import SearchConfig, SearchOverrides, SearchService
from footnote.domain.answer import AbstentionReason, Usage
from footnote.domain.documents import CollectionId
from footnote.domain.errors import ConflictError, UnavailableError, ValidationError
from footnote.domain.evidence import GateKind
from footnote.domain.settings import (
    AnswerSettings,
    CollectionSettings,
    RetrievalSettings,
    SearchMode,
)
from footnote.domain.tenancy import TenantId

from .fakes import (
    FakeClock,
    FakeCollection,
    FakeEmbedderFactory,
    FakeLLM,
    FakeReranker,
    FakeSearchStore,
    make_space,
)

TENANT = TenantId(UUID(int=7))
COLLECTION = CollectionId(UUID(int=99))
QUESTION = "beta gamma"
# Rerank logit = 2 - 1 = 1, so the rerank gate score is sigmoid(1) ~ 0.731.
LOW, HIGH = 0.5, 0.9


@dataclass
class Rig:
    service: AnswerService
    llm: FakeLLM
    reranker: FakeReranker | None
    collection: FakeCollection

    async def answer(self, question: str = QUESTION, **kw: Any) -> Any:
        mode = kw.pop("mode", SearchMode.LEXICAL)
        return await self.service.answer(
            TENANT, COLLECTION, AnswerRequest(question, mode=mode, **kw)
        )

    async def events(self, question: str = QUESTION, **kw: Any) -> list[AnswerEvent]:
        mode = kw.pop("mode", SearchMode.LEXICAL)
        request = AnswerRequest(question, mode=mode, **kw)
        return [e async for e in self.service.stream(TENANT, COLLECTION, request)]


def build(  # noqa: PLR0913 - test rig with explicit knobs
    llm_text: str = "A [1].",
    *,
    n_chunks: int = 8,
    rerank: bool = True,
    answer: AnswerSettings | None = None,
    config: AnswerConfig | None = None,
    calibrated: float | None = None,
    space_threshold: float | None = None,
    llm: FakeLLM | None = None,
) -> Rig:
    settings = CollectionSettings(
        retrieval=replace(RetrievalSettings(), rerank=rerank),
        answer=answer
        or AnswerSettings(evidence_threshold_rerank=LOW, evidence_threshold_vector=LOW),
    )
    collection = FakeCollection(make_space(space_threshold), settings)
    for n in range(n_chunks):
        collection.add(f"beta gamma passage{n}", title=f"Doc {n}")
    reranker = FakeReranker(calibrated_threshold=calibrated)
    search = SearchService(
        FakeSearchStore(collection),
        FakeEmbedderFactory(),
        reranker,
        FakeClock(),
        SearchConfig(),
    )
    llm = llm or FakeLLM(llm_text)
    return Rig(AnswerService(search, llm, config), llm, reranker, collection)


def _types(events: list[AnswerEvent]) -> list[str]:
    return [e.event for e in events]


async def test_answer_llm_not_configured_is_unavailable() -> None:
    rig = build(llm=FakeLLM("x", configured=False))
    with pytest.raises(UnavailableError) as info:
        await rig.answer()
    assert info.value.code == "llm_not_configured"


async def test_stream_llm_not_configured_is_unavailable() -> None:
    rig = build(llm=FakeLLM("x", configured=False))
    with pytest.raises(UnavailableError) as info:
        await rig.events()
    assert info.value.code == "llm_not_configured"


@pytest.mark.parametrize("question", ["", "   ", "x" * (QUESTION_MAX_CHARS + 1)])
async def test_answer_invalid_question_rejected(question: str) -> None:
    rig = build()
    with pytest.raises(ValidationError) as info:
        await rig.answer(question)
    assert info.value.field == "question"
    assert rig.llm.chat_calls == []


async def test_answer_question_at_max_length_accepted() -> None:
    rig = build()
    outcome = await rig.answer("beta " * 799 + "gamma")
    assert outcome.answer.abstained or outcome.answer.text is not None


@pytest.mark.parametrize(
    ("collection", "config", "calibrated", "passes"),
    [
        (0.1, 0.99, 0.99, True),
        (0.99, 0.1, 0.1, False),
        (None, 0.1, 0.99, True),
        (None, 0.99, 0.1, False),
        (None, None, 0.1, True),
        (None, None, 0.99, False),
    ],
)
async def test_answer_rerank_gate_resolution_order(
    collection: float | None, config: float | None, calibrated: float | None, passes: bool
) -> None:
    rig = build(
        answer=AnswerSettings(evidence_threshold_rerank=collection),
        config=AnswerConfig(evidence_threshold_rerank=config),
        calibrated=calibrated,
    )
    outcome = await rig.answer()
    assert outcome.gate.kind is GateKind.RERANK
    assert outcome.gate.passed is passes
    assert outcome.answer.abstained is (not passes)


async def test_answer_rerank_gate_without_any_threshold_conflicts() -> None:
    rig = build(answer=AnswerSettings())
    with pytest.raises(ConflictError) as info:
        await rig.answer()
    assert info.value.code == "evidence_threshold_uncalibrated"
    assert rig.llm.chat_calls == []


@pytest.mark.parametrize(
    ("collection", "config", "space", "passes"),
    [
        (0.5, 1.5, 1.5, True),
        (1.5, 0.5, 0.5, False),
        (None, 0.5, 1.5, True),
        (None, 1.5, 0.5, False),
        (None, None, 0.5, True),
        (None, None, 1.5, False),
    ],
)
async def test_answer_vector_gate_resolution_order(
    collection: float | None, config: float | None, space: float | None, passes: bool
) -> None:
    rig = build(
        rerank=False,
        answer=AnswerSettings(evidence_threshold_vector=collection),
        config=AnswerConfig(evidence_threshold_vector=config),
        space_threshold=space,
        calibrated=0.0,
    )
    # The question equals a chunk text, so the best cosine similarity is 1.
    rig.collection.add(QUESTION)
    outcome = await rig.answer(mode=SearchMode.HYBRID)
    assert outcome.gate.kind is GateKind.VECTOR
    assert outcome.gate.score == pytest.approx(1.0)
    assert outcome.gate.passed is passes


async def test_answer_vector_gate_without_any_threshold_conflicts() -> None:
    rig = build(rerank=False, answer=AnswerSettings())
    with pytest.raises(ConflictError) as info:
        await rig.answer()
    assert info.value.code == "evidence_threshold_uncalibrated"


async def test_answer_below_threshold_abstains_without_calling_llm() -> None:
    rig = build(answer=AnswerSettings(evidence_threshold_rerank=HIGH))
    outcome = await rig.answer()
    answer = outcome.answer
    assert answer.abstained
    assert answer.text is None
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert answer.gate_score == pytest.approx(0.731, abs=1e-3)
    assert rig.llm.chat_calls == []


async def test_answer_no_hits_abstains() -> None:
    rig = build(answer=AnswerSettings(evidence_threshold_rerank=0.0, evidence_threshold_vector=0.0))
    outcome = await rig.answer("nomatch")
    assert outcome.answer.abstained
    assert outcome.gate.score is None
    assert outcome.search.hits == []


async def test_answer_passing_gate_returns_verified_citations() -> None:
    rig = build("Beta is covered [1]. Gamma too [2].")
    outcome = await rig.answer()
    answer = outcome.answer
    assert not answer.abstained
    assert answer.text == "Beta is covered [1]. Gamma too [2]."
    assert [c.n for c in answer.citations] == [1, 2]
    assert answer.citations[0].chunk_id == outcome.search.hits[0].chunk_id
    assert answer.citations[0].title == outcome.search.hits[0].title
    assert not answer.unsupported
    assert answer.usage == Usage(11, 7)
    assert answer.gate_score == outcome.gate.score
    prompt = rig.llm.chat_calls[0]
    assert [m.role for m in prompt] == ["system", "user"]
    assert "[8]" in prompt[1].content
    assert "Question: beta gamma" in prompt[1].content


async def test_answer_removes_citation_beyond_hit_count() -> None:
    rig = build("A [1]. B [9].")
    outcome = await rig.answer()
    assert len(outcome.search.hits) == 8
    assert "[9]" not in (outcome.answer.text or "")
    assert [c.n for c in outcome.answer.citations] == [1]
    assert outcome.answer.text is not None
    assert outcome.answer.text.startswith("A [1].")


async def test_answer_strips_think_prefix() -> None:
    rig = build("<think>I should cite [3]</think>\nA [1].")
    outcome = await rig.answer()
    assert outcome.answer.text == "A [1]."
    assert [c.n for c in outcome.answer.citations] == [1]


async def test_answer_without_citation_is_unsupported() -> None:
    rig = build("Nothing cited here.")
    outcome = await rig.answer()
    assert outcome.answer.unsupported
    assert not outcome.answer.abstained
    assert outcome.answer.text == "Nothing cited here."
    assert outcome.answer.citations == ()


async def test_answer_unsupported_as_abstain_abstains_with_reason() -> None:
    answer_settings = AnswerSettings(evidence_threshold_rerank=LOW, unsupported_as_abstain=True)
    rig = build("Nothing cited here.", answer=answer_settings)
    outcome = await rig.answer()
    answer = outcome.answer
    assert answer.abstained
    assert answer.text is None
    assert answer.abstention_reason is AbstentionReason.UNSUPPORTED
    assert answer.unsupported
    assert answer.usage == Usage(11, 7)


async def test_answer_gate_does_not_call_llm() -> None:
    rig = build()
    result, decision = await rig.service.gate(
        TENANT, COLLECTION, AnswerRequest(QUESTION, mode=SearchMode.LEXICAL)
    )
    assert decision.passed
    assert result.reranked
    assert rig.llm.chat_calls == []


async def test_answer_gate_scores_with_an_override_reranker_file() -> None:
    rig = build(calibrated=HIGH)
    other = FakeReranker(calibrated_threshold=LOW)
    result, decision = await rig.service.gate(
        TENANT,
        COLLECTION,
        AnswerRequest(QUESTION, mode=SearchMode.LEXICAL),
        overrides=SearchOverrides(reranker=other),
    )
    assert result.reranked
    assert other.calls
    assert rig.reranker is not None
    assert rig.reranker.calls == []
    assert decision.kind is GateKind.RERANK
    assert decision.threshold == LOW  # the override file's calibrated value
    assert decision.passed
    assert rig.llm.chat_calls == []


async def test_stream_event_order_with_deltas_citations_and_done() -> None:
    rig = build("A [1]. B [2].", llm=FakeLLM("A [1]. B [2].", chunk_size=5))
    events = await rig.events()
    types = _types(events)
    assert types[0] == "retrieval"
    assert types[-2:] == ["citations", "done"]
    assert set(types[1:-2]) == {"delta"}
    assert len(types) > 4
    assert "".join(e.data["text"] for e in events if e.event == "delta") == "A [1]. B [2]."
    assert len(events[0].data["hits"]) == 8
    citations = events[-2].data
    assert [c.n for c in citations["citations"]] == [1, 2]
    assert citations["answer"] == "A [1]. B [2]."
    assert citations["unsupported"] is False


async def test_stream_done_carries_usage() -> None:
    rig = build(llm=FakeLLM("A [1].", usage=Usage(5, 3)))
    events = await rig.events()
    assert events[-1].data == {
        "usage": {"input_tokens": 5, "output_tokens": 3},
        "abstained": False,
        "abstention_reason": None,
        "unsupported": False,
    }


async def test_stream_abstention_emits_retrieval_empty_citations_done() -> None:
    rig = build(answer=AnswerSettings(evidence_threshold_rerank=HIGH))
    events = await rig.events()
    assert _types(events) == ["retrieval", "citations", "done"]
    assert events[1].data == {"citations": [], "answer": None, "unsupported": False}
    assert events[2].data["abstained"] is True
    assert events[2].data["abstention_reason"] is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert rig.llm.stream_calls == []


async def test_stream_chunk_size_one_matches_non_streamed_answer() -> None:
    raw = "<think>hmm</think> Beta [1]. Gamma [9]."
    streamed = build(llm=FakeLLM(raw, chunk_size=1))
    plain = build(llm=FakeLLM(raw))
    events = await streamed.events()
    outcome = await plain.answer()
    final = events[-2].data
    assert final["answer"] == outcome.answer.text
    assert final["citations"] == list(outcome.answer.citations)
    assert final["unsupported"] == outcome.answer.unsupported
    assert len([e for e in events if e.event == "delta"]) == len(raw)


async def test_stream_unsupported_as_abstain_reports_abstention() -> None:
    answer_settings = AnswerSettings(evidence_threshold_rerank=LOW, unsupported_as_abstain=True)
    rig = build("No cite.", answer=answer_settings)
    events = await rig.events()
    assert events[-2].data["answer"] is None
    assert events[-1].data["abstention_reason"] is AbstentionReason.UNSUPPORTED
    assert events[-1].data["unsupported"] is True


async def test_stream_llm_error_mid_stream_emits_error_event() -> None:
    llm = FakeLLM("A [1]. B [2].", chunk_size=4, fail_after=2)
    events = await build(llm=llm).events()
    assert _types(events) == ["retrieval", "delta", "delta", "error"]
    assert events[-1].data["code"] == "upstream_failed"
    assert events[-1].data["message"] == "stream broke"


async def test_stream_llm_domain_error_keeps_its_code() -> None:
    error = UnavailableError("down", code="llm_timeout")
    events = await build(llm=FakeLLM("abc", fail_after=0, error=error)).events()
    assert _types(events) == ["retrieval", "error"]
    assert events[-1].data["code"] == "llm_timeout"


async def test_stream_error_after_last_delta_is_still_reported() -> None:
    llm = FakeLLM("ab", chunk_size=10, fail_after=1)
    events = await build(llm=llm).events()
    assert _types(events) == ["retrieval", "delta", "error"]


async def test_answer_no_hits_with_rerank_uses_rerank_gate_only() -> None:
    rig = build(answer=AnswerSettings(evidence_threshold_rerank=0.0))
    outcome = await rig.answer("nomatch")
    assert outcome.answer.abstained
    assert outcome.gate.kind is GateKind.RERANK
    assert outcome.gate.score is None
