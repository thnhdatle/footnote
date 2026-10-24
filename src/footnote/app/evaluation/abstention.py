# SPDX-License-Identifier: Apache-2.0
"""Evaluates and calibrates the abstention gate on answerable and unanswerable claims."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import orjson

from footnote.app.answer import AnswerRequest
from footnote.app.collections import SettingsPatch
from footnote.app.evaluation.corpus import IndexedCorpus, index_corpus
from footnote.app.evaluation.datasets import BeirDataset, CorpusDoc, query_ids
from footnote.app.evaluation.manifest_edit import stored_threshold
from footnote.app.evaluation.metrics import CurvePoint, abstention_curve, rates_at, roc_auc
from footnote.app.evaluation.retrieval import EvalError, EvalServices
from footnote.app.search import SearchOverrides, SearchResult
from footnote.domain.documents import CollectionId
from footnote.domain.evidence import GateDecision, GateKind, calibrate_threshold
from footnote.domain.ports import Reranker
from footnote.domain.settings import ChunkingStrategyName, Language
from footnote.domain.tenancy import TenantId

ABSTENTION_PER_GROUP = 100
ABSTENTION_SPLIT = "test"
UNANSWERABLE = "unanswerable"
ANSWERABLE = "answerable"
OUT_OF_CORPUS_SET = Path("eval/sets/unanswerable.jsonl")

type Log = Callable[[str], None]


def _quiet(_message: str) -> None:
    return None


def gold_documents(qrels: Mapping[str, Mapping[str, int]]) -> dict[str, frozenset[str]]:
    return {q: frozenset(d for d, rel in docs.items() if rel > 0) for q, docs in qrels.items()}


def sha256_order(ids: Iterable[str]) -> list[str]:
    return sorted(ids, key=lambda q: (hashlib.sha256(q.encode("utf-8")).hexdigest(), q))


@dataclass(frozen=True)
class AbstentionSplit:
    unanswerable: tuple[str, ...]
    answerable: tuple[str, ...]
    skipped: tuple[str, ...]
    removed: frozenset[str]


def _other(group: str) -> str:
    return ANSWERABLE if group == UNANSWERABLE else UNANSWERABLE


def split_groups(
    ids: Sequence[str], gold: Mapping[str, frozenset[str]], per_group: int = ABSTENTION_PER_GROUP
) -> AbstentionSplit:
    groups: dict[str, list[str]] = {UNANSWERABLE: [], ANSWERABLE: []}
    docs: dict[str, set[str]] = {UNANSWERABLE: set(), ANSWERABLE: set()}
    skipped: list[str] = []
    turn = UNANSWERABLE
    for qid in sha256_order(q for q in ids if gold.get(q)):
        if all(len(g) >= per_group for g in groups.values()):
            break
        target = turn if len(groups[turn]) < per_group else _other(turn)
        claim = gold[qid]
        if claim & docs[_other(target)]:
            skipped.append(qid)
            continue
        groups[target].append(qid)
        docs[target] |= claim
        turn = _other(target)
    if any(len(g) < per_group for g in groups.values()):
        counts = ", ".join(f"{k} {len(v)}" for k, v in groups.items())
        raise EvalError(
            f"answerable/unanswerable split needs {per_group} claims per group; got {counts}"
        )
    return AbstentionSplit(
        tuple(groups[UNANSWERABLE]),
        tuple(groups[ANSWERABLE]),
        tuple(skipped),
        frozenset(docs[UNANSWERABLE]),
    )


def abstention_split(
    dataset: BeirDataset, per_group: int = ABSTENTION_PER_GROUP
) -> AbstentionSplit:
    if ABSTENTION_SPLIT not in dataset.qrels:
        raise EvalError(f"dataset {dataset.name} has no {ABSTENTION_SPLIT} qrels split")
    ids = [q for q in query_ids(dataset, ABSTENTION_SPLIT) if q in dataset.queries]
    return split_groups(ids, gold_documents(dataset.qrels[ABSTENTION_SPLIT]), per_group)


def index_documents(dataset: BeirDataset, split: AbstentionSplit) -> list[CorpusDoc]:
    return [d for d in dataset.corpus.values() if d.doc_id not in split.removed]


def calibration_claims(
    ids: Sequence[str], gold: Mapping[str, frozenset[str]], indexed: frozenset[str]
) -> list[str]:
    return [q for q in ids if gold.get(q) and gold[q] <= indexed]


@dataclass(frozen=True)
class GateIndex:
    tenant: TenantId
    collection_id: CollectionId
    corpus: IndexedCorpus
    documents: int
    removed: int


@asynccontextmanager
async def gate_index(  # noqa: PLR0913 - the indexing inputs
    services: EvalServices,
    dataset: BeirDataset,
    split: AbstentionSplit,
    *,
    run_id: str,
    chunking: ChunkingStrategyName,
    concurrency: int = 8,
    keep: bool = False,
    log: Log = _quiet,
) -> AsyncIterator[GateIndex]:
    """Indexes the abstention corpus into a temporary tenant."""
    docs = index_documents(dataset, split)
    tenant = (await services.admin.create_tenant(f"eval-gate-{dataset.name}-{run_id}")).id
    log(f"temporary tenant {tenant}")
    try:
        patch = SettingsPatch(chunking={"strategy": chunking}, language=Language.ENGLISH)
        change = await services.collections.create(tenant, f"{dataset.name}-abstention", patch)
        cid = change.collection.record.id
        removed = len(dataset.corpus) - len(docs)
        log(f"indexing {len(docs)} documents ({removed} unanswerable-group gold docs removed)")
        step = max(1, len(docs) // 10)

        def progress(done: int, total: int) -> None:
            if done % step == 0 or done == total:
                log(f"  {done}/{total}")

        corpus = await index_corpus(
            services.uploader,
            services.ingest,
            tenant,
            cid,
            docs,
            concurrency=concurrency,
            progress=progress,
        )
        log(f"indexed: {corpus.chunks} chunks, {corpus.seconds:.0f}s")
        yield GateIndex(tenant, cid, corpus, len(docs), removed)
    finally:
        if not keep:
            log(f"deleting temporary tenant {tenant}")
            await services.delete_tenant(tenant)


class Gate(Protocol):
    async def gate(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        request: AnswerRequest,
        *,
        overrides: SearchOverrides | None = None,
    ) -> tuple[SearchResult, GateDecision]: ...


@dataclass(frozen=True)
class GateSpec:
    kind: GateKind
    key: str
    sha256: str | None = None
    reranker: Reranker | None = None

    @property
    def label(self) -> str:
        return f"{self.kind.value} {self.key}"


@dataclass
class GateScores:
    spec: GateSpec
    scores: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None

    def of(self, ids: Iterable[str]) -> list[float]:
        return [self.scores[q] for q in ids]


async def score_gate(
    gate: Gate,
    index: GateIndex,
    questions: Sequence[tuple[str, str]],
    spec: GateSpec,
    *,
    log: Log = _quiet,
) -> GateScores:
    """Scores every question with the gate."""
    out = GateScores(spec)
    rerank = spec.kind is GateKind.RERANK
    overrides = SearchOverrides(reranker=spec.reranker) if spec.reranker is not None else None
    step = max(1, len(questions) // 5)
    for n, (qid, text) in enumerate(questions, start=1):
        _, decision = await gate.gate(
            index.tenant,
            index.collection_id,
            AnswerRequest(question=text, rerank=rerank),
            overrides=overrides,
        )
        if decision.kind is not spec.kind:
            raise EvalError(f"{spec.label}: the gate decided with the {decision.kind} score")
        out.scores[qid] = 0.0 if decision.score is None else decision.score
        out.threshold = decision.threshold
        if n % step == 0 or n == len(questions):
            log(f"  {spec.label}: {n}/{len(questions)}")
    return out


def load_question_set(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"question set {path} not found")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = orjson.loads(line)
            qid, text = str(row["id"]), str(row["text"]).strip()
        except (orjson.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvalError(f"{path}:{number}: expected an object with id and text") from exc
        if not text or qid in seen:
            raise EvalError(f"{path}:{number}: empty text or duplicate id {qid!r}")
        seen.add(qid)
        out.append((qid, text))
    if not out:
        raise EvalError(f"{path} has no questions")
    return out


@dataclass(frozen=True)
class GateSummary:
    auc: float
    threshold: float
    at_threshold: CurvePoint
    curve: list[CurvePoint]
    best: CurvePoint

    def as_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc,
            "threshold": self.threshold,
            "answer_rate": self.at_threshold.answer_rate,
            "abstain_rate": self.at_threshold.abstain_rate,
            "best_balanced": {
                "threshold": self.best.threshold,
                "answer_rate": self.best.answer_rate,
                "abstain_rate": self.best.abstain_rate,
            },
        }


def summarise(
    positive: Sequence[float], negative: Sequence[float], threshold: float
) -> GateSummary:
    curve = abstention_curve(positive, negative)
    best = max(curve, key=lambda p: (min(p.answer_rate, p.abstain_rate), -p.threshold))
    return GateSummary(
        roc_auc(positive, negative), threshold, rates_at(positive, negative, threshold), curve, best
    )


def curve_csv(curves: Mapping[str, Sequence[CurvePoint]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = ["threshold"]
    for prefix in curves:
        header += [f"{prefix}answer_rate", f"{prefix}abstain_rate"]
    writer.writerow(header)
    columns = list(curves.values())
    for i, point in enumerate(columns[0]):
        row = [f"{point.threshold:.2f}"]
        for curve in columns:
            row += [f"{curve[i].answer_rate:.4f}", f"{curve[i].abstain_rate:.4f}"]
        writer.writerow(row)
    return buf.getvalue()


def scores_csv(rows: Iterable[tuple[str, str, Mapping[str, float]]], gates: Sequence[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["group", "id", *gates])
    for group, qid, scores in rows:
        writer.writerow([group, qid, *(f"{scores[g]:.6f}" for g in gates)])
    return buf.getvalue()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Calibration:
    spec: GateSpec
    threshold: float
    stored: float
    claims: int
    answered: int
    target: float
    current: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.spec.kind.value,
            "key": self.spec.key,
            "sha256": self.spec.sha256,
            "threshold": self.threshold,
            "stored": self.stored,
            "claims": self.claims,
            "answered_at_stored": self.answered,
            "answer_rate_at_stored": self.answered / self.claims,
            "target_answer_rate": self.target,
            "current": self.current,
        }


def calibrate(
    scores: GateScores, ids: Sequence[str], target: float, current: float | None
) -> Calibration:
    """Computes the threshold that reaches the target answer rate."""
    values = scores.of(ids)
    exact = calibrate_threshold(values, target)
    stored = stored_threshold(exact)
    answered = sum(1 for v in values if v >= stored)
    return Calibration(scores.spec, exact, stored, len(values), answered, target, current)
