# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import io
from pathlib import Path

from footnote.app.evaluation.bm25 import Bm25Index, document_text, fuse_rankings
from footnote.app.evaluation.datasets import CorpusDoc
from footnote.app.evaluation.report import (
    CSV_COLUMNS,
    METRICS,
    ReportMeta,
    Row,
    ablation_markdown,
    csv_text,
    write_reports,
)


def _row(name: str, *, shipped: bool = False, note: str = "") -> Row:
    metrics = dict.fromkeys(METRICS, 0.5)
    metrics["ndcg_at_10"] = 0.71234
    return Row(
        name, "test", "vector", "heading", "a|b", metrics, 12.34, 56.78, 300, shipped, "", note
    )


def test_csv_has_one_line_per_row_with_fixed_columns() -> None:
    rows = list(csv.reader(io.StringIO(csv_text([_row("vector", shipped=True), _row("x")]))))
    assert tuple(rows[0]) == CSV_COLUMNS
    first = dict(zip(rows[0], rows[1], strict=True))
    assert first["name"] == "vector"
    assert first["shipped"] == "yes"
    assert first["ndcg_at_10"] == "0.7123"
    assert (first["p50_ms"], first["p95_ms"], first["queries"]) == ("12.3", "56.8", "300")
    assert dict(zip(rows[0], rows[2], strict=True))["shipped"] == ""


def test_markdown_marks_shipped_rows_and_lists_beir_references() -> None:
    text = ablation_markdown(
        [_row("vector", shipped=True), _row("bm25s", note="reference")],
        ReportMeta("run-1", "scifact", "abc123", ["model x"]),
    )
    assert "| vector **shipped** | test | a\\|b | 0.5000" in text
    assert "| bm25s (reference) |" in text
    assert "| BEIR BM25 | 0.665 |" in text
    assert "| BEIR BM25+CE | 0.688 |" in text
    assert "`abc123`" in text
    assert "- model x" in text
    other = ablation_markdown([_row("v")], ReportMeta("r", "tiny", "c"))
    assert "No published reference recorded for tiny." in other


def test_write_reports_creates_both_files(tmp_path: Path) -> None:
    csv_path, md_path = write_reports(tmp_path / "run", [_row("v")], ReportMeta("r", "d", "c"))
    assert csv_path.name == "retrieval.csv"
    assert md_path.name == "ablation.md"
    assert csv_path.read_text().startswith("name,split,")
    assert md_path.read_text().startswith("# Retrieval evaluation: d")


CORPUS = {
    "a": CorpusDoc("a", "Running shoes", "Shoes designed for runners and joggers."),
    "b": CorpusDoc("b", "Cooking", "The chef cooks pasta with tomato sauce."),
    "c": CorpusDoc("c", "", "Dogs run in the park while the cat sleeps."),
}


def test_bm25_ranks_with_stemming_and_stopwords() -> None:
    index = Bm25Index(CORPUS)
    assert index.search("runner")[0] == "a"
    assert index.search("cooked pasta") == ["b"]
    assert index.search("the and of") == []
    assert index.search("zebra") == []
    assert set(index.search("run", k=10)) == {"a", "c"}
    assert document_text(CORPUS["c"]) == CORPUS["c"].text


def test_bm25_run_times_each_query() -> None:
    ticks = iter([0.0, 0.5, 1.0, 1.25])
    run = Bm25Index(CORPUS).run({"q1": "pasta", "q2": "dogs"}, clock=lambda: next(ticks))
    assert run.rankings == {"q1": ["b"], "q2": ["c"]}
    assert run.seconds == {"q1": 0.5, "q2": 0.25}


def test_fuse_rankings_weighted_rrf_by_hand() -> None:
    fused = fuse_rankings([(["x", "y"], 1.0), (["z", "x"], 0.5)], rrf_k=60)
    assert fused == ["x", "y", "z"]
    tie = fuse_rankings([(["p"], 1.0), (["q"], 1.0)], rrf_k=60, tiebreak=["q", "p"])
    assert tie == ["q", "p"]
    assert fuse_rankings([(["b", "a"], 0.0)], rrf_k=60) == ["a", "b"]
    assert fuse_rankings([(["a", "a", "b"], 1.0)]) == ["a", "b"]
