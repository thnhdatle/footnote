# SPDX-License-Identifier: Apache-2.0
"""`footnote eval abstention` and `footnote eval calibrate`: the abstention gate on
answerable vs unanswerable claims.

Group split (`split_groups`): the SciFact test query ids (qrels order, those with at least
one gold document) are ordered by the hex SHA-256 of the id (ties by id). Walking that
order, the claim goes to the group whose turn it is (unanswerable first), or to the other
group once the turn's group holds `per_group` claims. A claim whose gold documents
intersect the gold documents already in the *other* group is skipped and does not use
the turn; an assigned claim passes the turn to the other group. The walk stops when both
groups hold `per_group` claims (EvalError if the split runs out first).

Index: the corpus minus every gold document of the unanswerable group, ingested into a
temporary tenant with the shipped chunking (`gate_index`). Answerable claims keep all
their gold documents (they never share one with the unanswerable group).

Gate scores come from `AnswerService.gate`, the app-layer gate of `/answers`, with the
collection's shipped retrieval settings and no LLM call: rerank gate = sigmoid of the
best cross-encoder logit over the reranked top `rerank_n`; vector gate = best cosine
between the query and the final hits. A question with no hits scores 0.0 (abstains).
Other reranker files are scored through the same gate with an eval-only override.

Calibration (`calibrate`): the train claims whose gold documents all lie in that index;
threshold = the largest t such that at least `eval.calibration_target_answer_rate` of
them score >= t (`domain.evidence.calibrate_threshold`), per reranker file and per space.
"""

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


# --- Answerable/unanswerable split and the calibration claims -----------------------------


def gold_documents(qrels: Mapping[str, Mapping[str, int]]) -> dict[str, frozenset[str]]:
    """Query id → documents with relevance > 0 (qrels order)."""
    return {q: frozenset(d for d, rel in docs.items() if rel > 0) for q, docs in qrels.items()}


def sha256_order(ids: Iterable[str]) -> list[str]:
    """Ids ordered by the hex SHA-256 of their UTF-8 bytes (ties by id)."""
    return sorted(ids, key=lambda q: (hashlib.sha256(q.encode("utf-8")).hexdigest(), q))


@dataclass(frozen=True)
class AbstentionSplit:
    """The two claim groups (assignment order), the skipped claims and the removed docs."""

    unanswerable: tuple[str, ...]
    answerable: tuple[str, ...]
    skipped: tuple[str, ...]
    removed: frozenset[str]


def _other(group: str) -> str:
    return ANSWERABLE if group == UNANSWERABLE else UNANSWERABLE


def split_groups(
    ids: Sequence[str], gold: Mapping[str, frozenset[str]], per_group: int = ABSTENTION_PER_GROUP
) -> AbstentionSplit:
    """The claim group split (module docstring); EvalError when the ids run out."""
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
    """`split_groups` over the dataset's test qrels (claims with a query text only)."""
    if ABSTENTION_SPLIT not in dataset.qrels:
        raise EvalError(f"dataset {dataset.name} has no {ABSTENTION_SPLIT} qrels split")
    ids = [q for q in query_ids(dataset, ABSTENTION_SPLIT) if q in dataset.queries]
    return split_groups(ids, gold_documents(dataset.qrels[ABSTENTION_SPLIT]), per_group)


def index_documents(dataset: BeirDataset, split: AbstentionSplit) -> list[CorpusDoc]:
    """The corpus minus the unanswerable group's gold documents (corpus order)."""
    return [d for d in dataset.corpus.values() if d.doc_id not in split.removed]


def calibration_claims(
    ids: Sequence[str], gold: Mapping[str, frozenset[str]], indexed: frozenset[str]
) -> list[str]:
    """Claims (given order) with at least one gold document, all of them in `indexed`."""
    return [q for q in ids if gold.get(q) and gold[q] <= indexed]


# --- the temporary index ----------------------------------------------------------------


@dataclass(frozen=True)
class GateIndex:
    """The collection the gate runs against, and what went into it."""

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
    """Index the abstention corpus into a temporary tenant (deleted on exit unless `keep`)."""
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


# --- gate scores ------------------------------------------------------------------------


class Gate(Protocol):
    """`AnswerService.gate`."""

    async def gate(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        request: AnswerRequest,
        *,
        overrides: SearchOverrides | None = None,
    ) -> tuple[SearchResult, GateDecision]:
        """Search and decide without calling the LLM."""


@dataclass(frozen=True)
class GateSpec:
    """One gate to score: the rerank gate with a reranker file, or the vector gate.

    `name` labels outputs (`rerank` / `vector`); `key` is the file key (`int8-arm64`,
    `fp32`, …) or the space name. `reranker` None = the service's configured reranker.
    """

    kind: GateKind
    key: str
    sha256: str | None = None
    reranker: Reranker | None = None

    @property
    def label(self) -> str:
        """`rerank <file>` or `vector <space>`."""
        return f"{self.kind.value} {self.key}"


@dataclass
class GateScores:
    """Scores per question id and the threshold the gate resolved (shipped config)."""

    spec: GateSpec
    scores: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None

    def of(self, ids: Iterable[str]) -> list[float]:
        """Scores of `ids`, in order."""
        return [self.scores[q] for q in ids]


async def score_gate(
    gate: Gate,
    index: GateIndex,
    questions: Sequence[tuple[str, str]],
    spec: GateSpec,
    *,
    log: Log = _quiet,
) -> GateScores:
    """Gate score of every question (sequential; 0.0 when the search has no hits)."""
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


# --- out-of-corpus questions ------------------------------------------------------------


def load_question_set(path: Path) -> list[tuple[str, str]]:
    """(id, text) from a JSONL file of `{"id", "text"}` objects; FileNotFoundError if absent."""
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


# --- abstention summary and outputs -----------------------------------------------------


@dataclass(frozen=True)
class GateSummary:
    """AUC (answerable positive), rates at a threshold and the full curve."""

    auc: float
    threshold: float
    at_threshold: CurvePoint
    curve: list[CurvePoint]
    best: CurvePoint

    def as_dict(self) -> dict[str, Any]:
        """JSON form (no curve)."""
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
    """Gate summary; `best` maximises min(answer, abstain) over the 0.01 grid."""
    curve = abstention_curve(positive, negative)
    best = max(curve, key=lambda p: (min(p.answer_rate, p.abstain_rate), -p.threshold))
    return GateSummary(
        roc_auc(positive, negative), threshold, rates_at(positive, negative, threshold), curve, best
    )


def curve_csv(curves: Mapping[str, Sequence[CurvePoint]]) -> str:
    """Thresholds 0.00..1.00 with `<prefix>answer_rate`, `<prefix>abstain_rate` per curve.

    The first curve's prefix is normally "" so the file starts with the columns
    `threshold,answer_rate,abstain_rate`.
    """
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
    """One row per question: group, id, one score column per gate."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["group", "id", *gates])
    for group, qid, scores in rows:
        writer.writerow([group, qid, *(f"{scores[g]:.6f}" for g in gates)])
    return buf.getvalue()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Pretty JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# --- calibration ------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Calibrated threshold of one gate (exact and stored) over the calibration claims."""

    spec: GateSpec
    threshold: float
    stored: float
    claims: int
    answered: int
    target: float
    current: float | None

    def as_dict(self) -> dict[str, Any]:
        """JSON form."""
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
    """Threshold for `scores` over `ids` at `target` answer rate (stored = 4 decimals, down)."""
    values = scores.of(ids)
    exact = calibrate_threshold(values, target)
    stored = stored_threshold(exact)
    answered = sum(1 for v in values if v >= stored)
    return Calibration(scores.spec, exact, stored, len(values), answered, target, current)
