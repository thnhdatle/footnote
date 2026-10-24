# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import io
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from footnote.app.answer import AnswerRequest
from footnote.app.evaluation.abstention import (
    ANSWERABLE,
    UNANSWERABLE,
    GateIndex,
    GateScores,
    GateSpec,
    abstention_split,
    calibrate,
    calibration_claims,
    curve_csv,
    gold_documents,
    index_documents,
    load_question_set,
    score_gate,
    scores_csv,
    sha256_order,
    split_groups,
    summarise,
)
from footnote.app.evaluation.corpus import IndexedCorpus
from footnote.app.evaluation.datasets import BeirDataset, CorpusDoc
from footnote.app.evaluation.retrieval import EvalError
from footnote.app.search import SearchOverrides
from footnote.domain.documents import CollectionId
from footnote.domain.evidence import GateDecision, GateKind
from footnote.domain.tenancy import TenantId

from .fakes import FakeReranker

TENANT = TenantId(UUID(int=7))
COLLECTION = CollectionId(UUID(int=9))


def _ordered(n: int) -> list[str]:
    return sha256_order(f"q{i}" for i in range(n))


def test_sha256_order_sorts_by_hex_digest() -> None:
    ids = ["3", "1", "2", "10"]
    expected = sorted(ids, key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    assert sha256_order(ids) == expected


def test_gold_documents_keeps_positive_relevance_only() -> None:
    assert gold_documents({"q": {"a": 1, "b": 0, "c": 2}}) == {"q": frozenset({"a", "c"})}


def test_split_groups_alternates_and_skips_claims_sharing_the_other_groups_docs() -> None:
    o = _ordered(10)
    gold = {
        o[0]: frozenset({"d0"}),
        o[1]: frozenset({"d1"}),
        o[2]: frozenset({"d0", "d2"}),
        o[3]: frozenset({"d0"}),
        o[4]: frozenset({"d4"}),
        o[5]: frozenset({"d1"}),
        o[6]: frozenset({"d6"}),
        o[7]: frozenset({"d7"}),
        o[8]: frozenset({"d8"}),
        o[9]: frozenset({"d9"}),
    }
    split = split_groups(list(reversed(o)), gold, per_group=3)
    assert split.unanswerable == (o[0], o[2], o[6])
    assert split.answerable == (o[1], o[4], o[7])
    assert split.skipped == (o[3], o[5])
    assert split.removed == frozenset({"d0", "d2", "d6"})


def test_split_groups_ignores_claims_without_gold_and_fails_when_short() -> None:
    o = _ordered(4)
    gold = {o[0]: frozenset({"a"}), o[1]: frozenset(), o[2]: frozenset({"a"})}
    with pytest.raises(EvalError, match="2 claims per group"):
        split_groups(o, gold, per_group=2)


@pytest.mark.parametrize("seed", range(20))
def test_split_groups_invariants_on_random_qrels(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    ids = [f"{rng.randrange(10**6)}-{i}" for i in range(120)]
    gold = {q: frozenset(f"d{rng.randrange(150)}" for _ in range(rng.randint(1, 3))) for q in ids}
    split = split_groups(ids, gold, per_group=30)
    assert len(split.unanswerable) == len(split.answerable) == 30
    u_docs = frozenset().union(*(gold[q] for q in split.unanswerable))
    a_docs = frozenset().union(*(gold[q] for q in split.answerable))
    assert not u_docs & a_docs
    assert split.removed == u_docs
    order = sha256_order(ids)
    used = [q for q in order if q in {*split.unanswerable, *split.answerable, *split.skipped}]
    assert used == order[: len(used)]


def _dataset() -> BeirDataset:
    o = _ordered(6)
    corpus = {f"d{i}": CorpusDoc(f"d{i}", f"T{i}", f"text {i}") for i in range(8)}
    test = {q: {f"d{i}": 1} for i, q in enumerate(o)}
    train = {"t1": {"d0": 1}, "t2": {"d1": 1, "d7": 1}, "t3": {"d7": 0}, "t4": {"d6": 1}}
    queries = {q: f"claim {q}" for q in [*o, *train]}
    return BeirDataset("tiny", corpus, queries, {"test": test, "train": train})


def test_abstention_split_index_and_calibration_claims_on_a_dataset() -> None:
    data = _dataset()
    split = abstention_split(data, per_group=3)
    o = _ordered(6)
    assert split.unanswerable == (o[0], o[2], o[4])
    assert split.removed == frozenset({"d0", "d2", "d4"})
    indexed = frozenset(d.doc_id for d in index_documents(data, split))
    assert indexed == frozenset({"d1", "d3", "d5", "d6", "d7"})
    gold = gold_documents(data.qrels["train"])
    assert calibration_claims(list(gold), gold, indexed) == ["t2", "t4"]


@dataclass
class FakeGate:
    scores: dict[str, float | None]
    kind: GateKind | None = None
    calls: list[tuple[AnswerRequest, SearchOverrides | None]] = field(default_factory=list)

    async def gate(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        request: AnswerRequest,
        *,
        overrides: SearchOverrides | None = None,
    ) -> tuple[Any, GateDecision]:
        assert (tenant, collection_id) == (TENANT, COLLECTION)
        self.calls.append((request, overrides))
        kind = self.kind or (GateKind.RERANK if request.rerank else GateKind.VECTOR)
        score = self.scores[request.question]
        return None, GateDecision(kind, score, 0.35, score is not None and score >= 0.35)


INDEX = GateIndex(TENANT, COLLECTION, IndexedCorpus({}, 0, 0, 0.0), 0, 0)


async def test_score_gate_uses_the_gate_and_scores_no_hits_as_zero() -> None:
    gate = FakeGate({"a": 0.9, "b": None})
    reranker = FakeReranker()
    spec = GateSpec(GateKind.RERANK, "fp32", "sha", reranker)
    out = await score_gate(gate, INDEX, [("1", "a"), ("2", "b")], spec)
    assert out.scores == {"1": 0.9, "2": 0.0}
    assert out.threshold == 0.35
    assert all(r.rerank is True for r, _ in gate.calls)
    assert all(o is not None and o.reranker is reranker for _, o in gate.calls)
    vector = await score_gate(gate, INDEX, [("1", "a")], GateSpec(GateKind.VECTOR, "space"))
    assert vector.scores == {"1": 0.9}
    assert gate.calls[-1][0].rerank is False
    assert gate.calls[-1][1] is None


async def test_score_gate_rejects_a_gate_of_the_other_kind() -> None:
    gate = FakeGate({"a": 0.5}, kind=GateKind.VECTOR)
    with pytest.raises(EvalError, match="vector score"):
        await score_gate(gate, INDEX, [("1", "a")], GateSpec(GateKind.RERANK, "int8-arm64"))


def _scores(values: Sequence[float]) -> GateScores:
    out = GateScores(GateSpec(GateKind.RERANK, "int8-arm64", "sha"))
    out.scores = {f"c{i}": v for i, v in enumerate(values)}
    return out


def test_calibrate_is_the_largest_threshold_reaching_the_target_rate() -> None:
    values = [0.95, 0.9, 0.8, 0.7, 0.66666666, 0.6, 0.5, 0.4, 0.3, 0.1]
    scores = _scores(values)
    result = calibrate(scores, list(scores.scores), 0.85, current=0.35)
    assert result.threshold == 0.3
    assert result.stored == 0.3
    assert result.answered == 9
    assert result.claims == 10
    assert result.as_dict()["answer_rate_at_stored"] == 0.9
    assert result.as_dict()["current"] == 0.35
    eight = calibrate(scores, list(scores.scores)[:8], 0.5, current=None)
    assert eight.threshold == 0.7
    rounded = calibrate(scores, ["c4", "c0"], 1.0, current=None)
    assert rounded.threshold == 0.66666666
    assert rounded.stored == 0.6666
    assert rounded.answered == 2


def test_summarise_auc_rates_and_best_balanced_threshold() -> None:
    positive = [0.9, 0.8, 0.7, 0.2]
    negative = [0.1, 0.3, 0.75, 0.05]
    s = summarise(positive, negative, 0.35)
    assert s.auc == pytest.approx(13 / 16)
    assert (s.at_threshold.answer_rate, s.at_threshold.abstain_rate) == (0.75, 0.75)
    assert len(s.curve) == 101
    assert min(s.best.answer_rate, s.best.abstain_rate) == 0.75
    assert s.as_dict()["best_balanced"]["threshold"] == s.best.threshold


def test_curve_csv_has_101_thresholds_and_the_spec_columns_first() -> None:
    a = summarise([0.9, 0.2], [0.1, 0.5], 0.3).curve
    b = summarise([0.6], [0.4], 0.3).curve
    rows = list(csv.reader(io.StringIO(curve_csv({"": a, "vector_": b}))))
    assert rows[0] == [
        "threshold",
        "answer_rate",
        "abstain_rate",
        "vector_answer_rate",
        "vector_abstain_rate",
    ]
    assert len(rows) == 102
    assert rows[1] == ["0.00", "1.0000", "0.0000", "1.0000", "0.0000"]
    assert rows[31] == ["0.30", "0.5000", "0.5000", "1.0000", "0.0000"]
    assert rows[-1][0] == "1.00"


def test_scores_csv_rows() -> None:
    text = scores_csv([(ANSWERABLE, "1", {"rerank": 0.5, "vector": 0.25})], ["rerank", "vector"])
    assert text == "group,id,rerank,vector\nanswerable,1,0.500000,0.250000\n"
    assert UNANSWERABLE == "unanswerable"


def test_load_question_set(tmp_path: Path) -> None:
    path = tmp_path / "unanswerable.jsonl"
    with pytest.raises(FileNotFoundError):
        load_question_set(path)
    path.write_text('{"id": "u1", "text": " What is X? "}\n\n{"id": "u2", "text": "Y?"}\n')
    assert load_question_set(path) == [("u1", "What is X?"), ("u2", "Y?")]
    path.write_text('{"id": "u1"}\n')
    with pytest.raises(EvalError, match=":1: expected"):
        load_question_set(path)
    path.write_text('{"id": "u1", "text": "a"}\n{"id": "u1", "text": "b"}\n')
    with pytest.raises(EvalError, match="duplicate"):
        load_question_set(path)
    path.write_text("\n")
    with pytest.raises(EvalError, match="no questions"):
        load_question_set(path)
