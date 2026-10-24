# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from footnote.app.answer import AnswerConfig
from footnote.domain.errors import ConflictError
from footnote.domain.evidence import GateKind
from footnote.domain.settings import AnswerSettings, SearchMode

from .test_answer import QUESTION, build


async def _vector_threshold(**kw: object) -> float:
    rig = build(rerank=False, **kw)  # type: ignore[arg-type]
    rig.collection.add(QUESTION)
    outcome = await rig.answer(mode=SearchMode.HYBRID)
    assert outcome.gate.kind is GateKind.VECTOR
    return outcome.gate.threshold


async def _rerank_threshold(**kw: object) -> float:
    outcome = await build(**kw).answer()  # type: ignore[arg-type]
    assert outcome.gate.kind is GateKind.RERANK
    return outcome.gate.threshold


async def test_vector_gate_each_level_wins_in_order() -> None:
    both = {"config": AnswerConfig(evidence_threshold_vector=0.2), "space_threshold": 0.3}
    assert (
        await _vector_threshold(answer=AnswerSettings(evidence_threshold_vector=0.1), **both) == 0.1
    )
    assert await _vector_threshold(answer=AnswerSettings(), **both) == 0.2
    assert await _vector_threshold(answer=AnswerSettings(), space_threshold=0.3) == 0.3


async def test_vector_gate_ignores_reranker_calibration() -> None:
    value = await _vector_threshold(answer=AnswerSettings(), space_threshold=0.3, calibrated=0.9)
    assert value == 0.3


async def test_vector_gate_uncalibrated_space_conflicts_even_with_calibrated_reranker() -> None:
    rig = build(rerank=False, answer=AnswerSettings(), calibrated=0.9)
    with pytest.raises(ConflictError) as info:
        await rig.answer(mode=SearchMode.HYBRID)
    assert info.value.code == "evidence_threshold_uncalibrated"
    assert rig.llm.chat_calls == []


async def test_vector_gate_zero_collection_threshold_is_respected() -> None:
    value = await _vector_threshold(
        answer=AnswerSettings(evidence_threshold_vector=0.0), space_threshold=0.3
    )
    assert value == 0.0


async def test_rerank_gate_each_level_wins_in_order() -> None:
    cfg = AnswerConfig(evidence_threshold_rerank=0.2)
    assert (
        await _rerank_threshold(
            answer=AnswerSettings(evidence_threshold_rerank=0.1), config=cfg, calibrated=0.3
        )
        == 0.1
    )
    assert await _rerank_threshold(answer=AnswerSettings(), config=cfg, calibrated=0.3) == 0.2
    assert await _rerank_threshold(answer=AnswerSettings(), calibrated=0.3) == 0.3


async def test_rerank_gate_ignores_space_threshold_and_vector_settings() -> None:
    rig = build(
        answer=AnswerSettings(evidence_threshold_vector=0.1),
        config=AnswerConfig(evidence_threshold_vector=0.2),
        space_threshold=0.3,
    )
    with pytest.raises(ConflictError) as info:
        await rig.answer()
    assert info.value.code == "evidence_threshold_uncalibrated"
    assert rig.llm.chat_calls == []
