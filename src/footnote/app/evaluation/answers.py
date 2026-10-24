# SPDX-License-Identifier: Apache-2.0
"""Evaluates faithfulness and citation precision of generated answers."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from footnote.app.answer import AnswerRequest, AnswerService
from footnote.app.collections import SettingsPatch
from footnote.app.evaluation.corpus import index_corpus
from footnote.app.evaluation.datasets import BeirDataset, query_ids
from footnote.app.evaluation.retrieval import EvalError
from footnote.domain.answer import AbstentionReason, Answer
from footnote.domain.citations import find_markers, split_sentences, strip_reasoning
from footnote.domain.documents import CollectionId
from footnote.domain.ports import LLM, ChatMessage
from footnote.domain.retrieval import Hit
from footnote.domain.settings import Language
from footnote.domain.tenancy import TenantId

if TYPE_CHECKING:
    from footnote.app.evaluation.runtime import EvalRuntime

QUESTION_TEMPLATE = "Is the following claim supported by the documents? Explain with citations: "
PROMPT_PATH = Path("eval/prompts/faithfulness.md")
SPLIT = "test"
LABELS = ("supported", "unsupported", "non_factual")
SAME_MODEL_NOTE = "Judge model = generator model (same-model judging inflates scores)"
ANSWERS_CSV = "answers.csv"
SUMMARY_MD = "summary.md"
CSV_COLUMNS = (
    "query_id",
    "abstained",
    "unsupported",
    "n_sentences",
    "supported",
    "non_factual",
    "citations",
    "citations_supported",
    "parse_failure",
)
JUDGE_ATTEMPTS = 2
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

type Log = Callable[[str], None]


def _quiet(_message: str) -> None:
    return None


@dataclass(frozen=True, slots=True)
class Generated:
    answer: Answer
    hits: Sequence[Hit]


class Generator(Protocol):
    async def generate(self, question: str) -> Generated: ...


class AnswerGenerator:
    def __init__(self, service: AnswerService, tenant: TenantId, collection: CollectionId) -> None:
        self._service = service
        self._tenant = tenant
        self._collection = collection

    async def generate(self, question: str) -> Generated:
        outcome = await self._service.answer(
            self._tenant, self._collection, AnswerRequest(question)
        )
        return Generated(outcome.answer, outcome.search.hits)


@dataclass(frozen=True, slots=True)
class SentenceVerdict:
    label: str
    supports: dict[int, bool]


@dataclass(frozen=True, slots=True)
class ItemResult:
    query_id: str
    abstained: bool
    unsupported: bool = False
    n_sentences: int = 0
    supported: int = 0
    non_factual: int = 0
    citations: int = 0
    citations_supported: int = 0
    parse_failure: bool = False


@dataclass(frozen=True, slots=True)
class AnswerEvalResult:
    items: list[ItemResult]
    generator_model: str
    judge_model: str
    faithfulness: float | None
    citation_precision: float | None
    answer_rate: float
    answered: int
    abstained: int
    unsupported: int
    non_factual: int
    sentences: int
    judge_parse_failures: int
    out_dir: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def same_model(self) -> bool:
        return self.judge_model == self.generator_model


@dataclass(frozen=True, slots=True)
class _Prepared:
    query_id: str
    sentences: list[str]
    cited: list[list[int]]
    chunks: dict[int, str]
    unsupported: bool


def question_for(claim: str) -> str:
    return f"{QUESTION_TEMPLATE}{claim}"


def select_items(dataset: BeirDataset, limit: int) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for qid in query_ids(dataset, SPLIT)[:limit]:
        text = dataset.queries.get(qid)
        if text is None:
            raise EvalError(f"query {qid} of qrels/{SPLIT}.tsv is not in queries.jsonl")
        items.append((qid, text))
    return items


def _prepare(query_id: str, generated: Generated) -> _Prepared:
    text = generated.answer.text or ""
    sentences: list[str] = []
    cited: list[list[int]] = []
    chunks: dict[int, str] = {}
    for span in split_sentences(text):
        sentence = text[span.start : span.end]
        numbers = [
            n for n in dict.fromkeys(find_markers(sentence)) if 1 <= n <= len(generated.hits)
        ]
        sentences.append(sentence)
        cited.append(numbers)
        for n in numbers:
            chunks[n] = generated.hits[n - 1].text
    return _Prepared(query_id, sentences, cited, chunks, generated.answer.unsupported)


def render_judge_input(prepared: _Prepared) -> str:
    lines = ["Answer sentences:", ""]
    for i, (sentence, numbers) in enumerate(
        zip(prepared.sentences, prepared.cited, strict=True), 1
    ):
        lines.append(f"{i}. {sentence}")
        if numbers:
            lines.append("   Cited chunks:")
            lines.extend(f"   [{n}] {prepared.chunks[n]}" for n in numbers)
        else:
            lines.append("   Cited chunks: none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_json(text: str) -> Any:
    body = strip_reasoning(text).strip()
    fence = _FENCE_RE.search(body)
    if fence:
        body = fence.group(1)
    start, end = body.find("{"), body.rfind("}")
    for candidate in (body, body[start : end + 1] if 0 <= start < end else ""):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _verdict(entry: object) -> tuple[int, SentenceVerdict] | None:
    if not isinstance(entry, dict):
        return None
    index, label, raw = entry.get("i"), entry.get("label"), entry.get("citations", [])
    if not isinstance(index, int) or isinstance(index, bool) or label not in LABELS:
        return None
    if not isinstance(raw, list):
        return None
    supports: dict[int, bool] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("n"), int):
            return None
        supports[item["n"]] = item.get("supports") is True
    return index, SentenceVerdict(label, supports)


def parse_judgement(text: str, cited: Sequence[Sequence[int]]) -> list[SentenceVerdict] | None:
    data = _load_json(text)
    entries = data.get("sentences") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    by_index: dict[int, SentenceVerdict] = {}
    for entry in entries:
        parsed = _verdict(entry)
        if parsed is None:
            return None
        by_index.setdefault(parsed[0], parsed[1])
    if any(i not in by_index for i in range(1, len(cited) + 1)):
        return None
    return [by_index[i] for i in range(1, len(cited) + 1)]


def score_item(
    prepared: _Prepared, verdicts: list[SentenceVerdict] | None, *, parse_failure: bool = False
) -> ItemResult:
    supported = non_factual = cites = cites_ok = 0
    for i, numbers in enumerate(prepared.cited):
        cites += len(numbers)
        if verdicts is None:
            continue
        verdict = verdicts[i]
        ok = sum(1 for n in numbers if verdict.supports.get(n, False))
        cites_ok += ok
        if verdict.label == "non_factual":
            non_factual += 1
        elif verdict.label == "supported" and ok > 0:
            supported += 1
    return ItemResult(
        prepared.query_id,
        False,
        prepared.unsupported,
        len(prepared.sentences),
        supported,
        non_factual,
        cites,
        cites_ok,
        parse_failure,
    )


async def judge_item(judge: LLM, rubric: str, prepared: _Prepared) -> ItemResult:
    if not prepared.sentences:
        return score_item(prepared, [])
    messages = [ChatMessage("system", rubric), ChatMessage("user", render_judge_input(prepared))]
    for _ in range(JUDGE_ATTEMPTS):
        reply = await judge.chat(messages)
        verdicts = parse_judgement(reply.text, prepared.cited)
        if verdicts is not None:
            return score_item(prepared, verdicts)
    return score_item(prepared, None, parse_failure=True)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _unsupported(generated: Generated) -> bool:
    answer = generated.answer
    return answer.unsupported or answer.abstention_reason is AbstentionReason.UNSUPPORTED


async def run_answer_eval(  # noqa: PLR0913 - the run inputs
    generator: Generator,
    judge: LLM,
    items: Sequence[tuple[str, str]],
    *,
    generator_model: str,
    judge_model: str,
    out_dir: Path | None = None,
    rubric: str | None = None,
    notes: Sequence[str] = (),
    log: Log = _quiet,
) -> AnswerEvalResult:
    """Generates every answer, judges the answered ones and writes the outputs."""
    rubric_text = rubric if rubric is not None else PROMPT_PATH.read_text(encoding="utf-8")
    generated: list[tuple[str, Generated]] = []
    for n, (qid, claim) in enumerate(items, 1):
        generated.append((qid, await generator.generate(question_for(claim))))
        log(f"generated {n}/{len(items)}")
    results: list[ItemResult] = []
    unsupported = 0
    for n, (qid, gen) in enumerate(generated, 1):
        unsupported += _unsupported(gen)
        if gen.answer.abstained:
            results.append(ItemResult(qid, True, _unsupported(gen)))
            continue
        results.append(await judge_item(judge, rubric_text, _prepare(qid, gen)))
        log(f"judged {n}/{len(generated)}")
    answered = [r for r in results if not r.abstained]
    sentences = sum(r.n_sentences for r in answered)
    non_factual = sum(r.non_factual for r in answered)
    result = AnswerEvalResult(
        items=results,
        generator_model=generator_model,
        judge_model=judge_model,
        faithfulness=_ratio(sum(r.supported for r in answered), sentences - non_factual),
        citation_precision=_ratio(
            sum(r.citations_supported for r in answered), sum(r.citations for r in answered)
        ),
        answer_rate=len(answered) / len(results) if results else 0.0,
        answered=len(answered),
        abstained=len(results) - len(answered),
        unsupported=unsupported,
        non_factual=non_factual,
        sentences=sentences,
        judge_parse_failures=sum(r.parse_failure for r in answered),
        out_dir=out_dir,
        notes=list(notes),
    )
    if out_dir is not None:
        write_outputs(out_dir, result)
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_outputs(out_dir: Path, result: AnswerEvalResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ANSWERS_CSV).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for r in result.items:
            writer.writerow(
                [
                    r.query_id,
                    int(r.abstained),
                    int(r.unsupported),
                    r.n_sentences,
                    r.supported,
                    r.non_factual,
                    r.citations,
                    r.citations_supported,
                    int(r.parse_failure),
                ]
            )
    lines = [
        "# Answer evaluation",
        "",
        f"- generator model: `{result.generator_model}`",
        f"- judge model: `{result.judge_model}`",
    ]
    if result.same_model:
        lines.append(f"- {SAME_MODEL_NOTE}")
    lines += [
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| faithfulness | {_fmt(result.faithfulness)} |",
        f"| citation precision | {_fmt(result.citation_precision)} |",
        f"| answer rate | {result.answer_rate:.4f} |",
        f"| items | {len(result.items)} |",
        f"| answered | {result.answered} |",
        f"| abstained | {result.abstained} |",
        f"| unsupported | {result.unsupported} |",
        f"| sentences (answered items) | {result.sentences} |",
        f"| non-factual sentences | {result.non_factual} |",
        f"| judge parse failures | {result.judge_parse_failures} |",
        "",
        "Metrics are over answered items only. Faithfulness = sentences the judge marks "
        "supported by their cited chunks / sentences not labelled non-factual; citation "
        "precision = citations whose chunk supports their sentence / all citations.",
    ]
    if result.notes:
        lines += ["", *result.notes]
    (out_dir / SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


async def eval_answers(  # noqa: PLR0913 - the run inputs
    rt: EvalRuntime,
    dataset: BeirDataset,
    *,
    limit: int,
    judge: LLM,
    judge_model: str,
    out_dir: Path,
    run_id: str,
    keep: bool = False,
    concurrency: int = 8,
    log: Log = _quiet,
    generator: Callable[[TenantId, CollectionId], Generator] | None = None,
) -> AnswerEvalResult:
    """Indexes the dataset and runs the answer evaluation."""
    items = select_items(dataset, limit)
    if not items:
        raise EvalError(f"dataset {dataset.name!r} has no qrels/{SPLIT}.tsv queries")
    services = rt.services
    tenant = (await services.admin.create_tenant(f"eval-answers-{dataset.name}-{run_id}")).id
    log(f"temporary tenant {tenant}")
    try:
        patch = SettingsPatch(chunking={"strategy": rt.shipped.chunking}, language=Language.ENGLISH)
        change = await services.collections.create(tenant, f"{dataset.name}-answers", patch)
        cid = change.collection.record.id
        docs = list(dataset.corpus.values())
        log(f"indexing {len(docs)} documents")
        corpus = await index_corpus(
            services.uploader, services.ingest, tenant, cid, docs, concurrency=concurrency
        )
        log(f"indexed {corpus.chunks} chunks in {corpus.seconds:.0f}s")
        make: Callable[[TenantId, CollectionId], Generator] = generator or (
            lambda t, c: AnswerGenerator(rt.runtime.api.answers, t, c)
        )
        return await run_answer_eval(
            make(tenant, cid),
            judge,
            items,
            generator_model=rt.settings.llm.model,
            judge_model=judge_model,
            out_dir=out_dir,
            notes=[f"dataset `{dataset.name}`, first {len(items)} test claims; run `{run_id}`"],
            log=log,
        )
    finally:
        if not keep:
            log(f"deleting temporary tenant {tenant}")
            await services.delete_tenant(tenant)
