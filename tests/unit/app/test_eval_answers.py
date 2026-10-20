# SPDX-License-Identifier: Apache-2.0
"""`eval answers`: item selection, judge parsing and retry, metric arithmetic, outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest

from footnote.app.evaluation.answers import (
    QUESTION_TEMPLATE,
    SAME_MODEL_NOTE,
    AnswerEvalResult,
    Generated,
    parse_judgement,
    question_for,
    run_answer_eval,
    select_items,
)
from footnote.app.evaluation.datasets import BeirDataset
from footnote.domain.answer import AbstentionReason, Answer
from footnote.domain.documents import ChunkId, DocumentId
from footnote.domain.ports import ChatMessage, ChatResult
from footnote.domain.retrieval import Hit

RUBRIC = "RUBRIC"
UNANSWERABLE = Path(__file__).parents[3] / "eval" / "sets" / "unanswerable.jsonl"
PROMPT = Path(__file__).parents[3] / "eval" / "prompts" / "faithfulness.md"


def hit(n: int) -> Hit:
    return Hit(
        ChunkId(UUID(int=n)),
        DocumentId(UUID(int=n)),
        f"T{n}",
        (),
        f"chunk text {n}",
        "",
        None,
        None,
    )


def answered(text: str, hits: int = 3) -> Generated:
    return Generated(Answer(text=text), [hit(i) for i in range(1, hits + 1)])


def abstained() -> Generated:
    return Generated(
        Answer(None, abstained=True, abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE), []
    )


class FakeGenerator:
    def __init__(self, outputs: dict[str, Generated]) -> None:
        self.outputs = outputs
        self.questions: list[str] = []

    async def generate(self, question: str) -> Generated:
        self.questions.append(question)
        return self.outputs[question.removeprefix(QUESTION_TEMPLATE)]


class FakeJudge:
    """Replies from a script, one reply per call, recording the messages."""

    configured = True

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(self.replies[len(self.calls) - 1], "judge")

    def stream(self, messages: Sequence[ChatMessage]) -> object:
        raise NotImplementedError


def verdicts(*sentences: tuple[str, dict[int, bool]]) -> str:
    return json.dumps(
        {
            "sentences": [
                {
                    "i": i,
                    "label": label,
                    "citations": [{"n": n, "supports": s} for n, s in cites.items()],
                }
                for i, (label, cites) in enumerate(sentences, 1)
            ]
        }
    )


async def run(
    gen: dict[str, Generated],
    replies: Sequence[str],
    *,
    judge_model: str = "judge",
    out_dir: Path | None = None,
) -> tuple[AnswerEvalResult, FakeJudge, FakeGenerator]:
    generator = FakeGenerator(gen)
    judge = FakeJudge(replies)
    result = await run_answer_eval(
        generator,
        judge,
        [(q, q) for q in gen],
        generator_model="gen",
        judge_model=judge_model,
        out_dir=out_dir,
        rubric=RUBRIC,
    )
    return result, judge, generator


def test_select_items_first_distinct_query_ids_in_file_order() -> None:
    dataset = BeirDataset(
        "d",
        {},
        {"b": "claim b", "a": "claim a", "c": "claim c"},
        {"test": {"b": {}, "a": {}, "c": {}}},
    )
    assert select_items(dataset, 2) == [("b", "claim b"), ("a", "claim a")]
    assert question_for("x") == (
        "Is the following claim supported by the documents? Explain with citations: x"
    )


async def test_metric_arithmetic_over_answered_items_only() -> None:
    gen = {
        "q1": answered("Alpha is true [1]. Beta is true [2][3]. I cannot say more."),
        "q2": answered("Gamma holds [1]. Delta holds."),
        "q3": abstained(),
    }
    replies = [
        verdicts(
            ("supported", {1: True}),
            ("supported", {2: True, 3: False}),
            ("non_factual", {}),
        ),
        verdicts(("unsupported", {1: False}), ("supported", {})),
    ]
    result, judge, generator = await run(gen, replies)
    assert len(judge.calls) == 2  # one judge call per answered item
    assert all(q.startswith(QUESTION_TEMPLATE) for q in generator.questions)
    # sentences not non-factual: 2 + 2 = 4; supported: 2 (the uncited "supported" is not).
    assert result.faithfulness == pytest.approx(2 / 4)
    # citations: 3 + 1 = 4 of which 2 support.
    assert result.citation_precision == pytest.approx(2 / 4)
    assert result.answer_rate == pytest.approx(2 / 3)
    assert (result.answered, result.abstained, result.non_factual) == (2, 1, 1)
    assert result.judge_parse_failures == 0


async def test_judge_sees_sentences_with_cited_chunks_only() -> None:
    gen = {"q": answered("Alpha [2]. Beta.")}
    _, judge, _ = await run(gen, [verdicts(("supported", {2: True}), ("unsupported", {}))])
    system, user = judge.calls[0]
    assert system.content == RUBRIC
    assert "1. Alpha [2]." in user.content
    assert "[2] chunk text 2" in user.content
    assert "chunk text 1" not in user.content
    assert "Cited chunks: none" in user.content


async def test_invalid_json_is_retried_once_and_fences_and_reasoning_are_tolerated() -> None:
    good = verdicts(("supported", {1: True}))
    replies = ["not json at all", "<think>hmm</think>\n```json\n" + good + "\n```"]
    result, judge, _ = await run({"q": answered("Alpha [1].")}, replies)
    assert len(judge.calls) == 2
    assert result.faithfulness == 1.0
    assert result.judge_parse_failures == 0


async def test_second_invalid_reply_counts_unsupported_and_a_parse_failure(tmp_path: Path) -> None:
    gen = {"q1": answered("Alpha [1]. Beta [2]."), "q2": answered("Gamma [1].")}
    replies = ["{", '{"sentences": "no"}', verdicts(("supported", {1: True}))]
    result, judge, _ = await run(gen, replies, out_dir=tmp_path)
    assert len(judge.calls) == 3
    assert result.judge_parse_failures == 1
    # q1: 2 unsupported sentences, 2 citations unsupported; q2: 1 supported.
    assert result.faithfulness == pytest.approx(1 / 3)
    assert result.citation_precision == pytest.approx(1 / 3)
    rows = list(csv.DictReader((tmp_path / "answers.csv").open(encoding="utf-8")))
    assert [r["parse_failure"] for r in rows] == ["1", "0"]
    assert rows[0]["n_sentences"] == "2"
    assert rows[0]["citations"] == "2"


def test_parse_rejects_missing_sentences_and_bad_labels() -> None:
    cited = [[1], []]
    assert parse_judgement(verdicts(("supported", {1: True})), cited) is None
    assert parse_judgement(verdicts(("fine", {1: True}), ("supported", {})), cited) is None
    parsed = parse_judgement(verdicts(("supported", {1: True}), ("non_factual", {})), cited)
    assert parsed is not None
    assert parsed[1].label == "non_factual"


async def test_non_factual_sentences_leave_the_faithfulness_denominator() -> None:
    gen = {"q": answered("The documents do not say. Alpha [1].")}
    result, _, _ = await run(gen, [verdicts(("non_factual", {}), ("supported", {1: True}))])
    assert result.faithfulness == 1.0
    assert result.non_factual == 1


async def test_all_non_factual_gives_no_faithfulness() -> None:
    result, _, _ = await run({"q": answered("Cannot find it.")}, [verdicts(("non_factual", {}))])
    assert result.faithfulness is None
    assert result.citation_precision is None


async def test_unsupported_answers_are_counted() -> None:
    flagged = Generated(Answer(text="Alpha.", unsupported=True), [hit(1)])
    result, _, _ = await run({"q": flagged}, [verdicts(("supported", {}))])
    assert result.unsupported == 1
    assert result.faithfulness == 0.0


@pytest.mark.parametrize(("judge", "same"), [("gen", True), ("other", False)])
async def test_same_model_note(tmp_path: Path, judge: str, same: bool) -> None:
    await run(
        {"q": answered("A [1].")},
        [verdicts(("supported", {1: True}))],
        judge_model=judge,
        out_dir=tmp_path,
    )
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert (SAME_MODEL_NOTE in summary) is same
    assert "faithfulness" in summary
    assert "`gen`" in summary


def test_rubric_prompt_names_the_labels() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert all(label in text for label in ("supported", "unsupported", "non_factual", "sentences"))


def test_unanswerable_set_has_200_unique_ids_and_texts() -> None:
    rows = [json.loads(line) for line in UNANSWERABLE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 200
    assert len({r["id"] for r in rows}) == 200
    assert len({r["text"].strip().lower() for r in rows}) == 200
    assert all(set(r) == {"id", "text"} and r["text"].strip() for r in rows)
