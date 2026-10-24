# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from footnote.app.evaluation.datasets import dataset_dir, load_beir, query_ids


def make_beir(root: Path) -> Path:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        '{"_id": "d2", "title": "T2", "text": "second"}\n'
        '{"_id": "d1", "title": "", "text": "first"}\n'
        '{"_id": "d3", "text": "third"}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "one", "metadata": {}}\n{"_id": "q2", "text": "two"}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq2\td1\t1\nq1\td2\t2\nq2\td3\t0\nq1\td1\t1\n",
        encoding="utf-8",
    )
    (root / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td3\t1\n")
    return root


def test_load_beir_reads_corpus_queries_and_qrels(tmp_path: Path) -> None:
    ds = load_beir(make_beir(tmp_path / "tiny"))
    assert ds.name == "tiny"
    assert list(ds.corpus) == ["d2", "d1", "d3"]
    assert ds.corpus["d2"].title == "T2"
    assert ds.corpus["d3"].title == ""
    assert ds.queries == {"q1": "one", "q2": "two"}
    assert ds.qrels["test"] == {"q2": {"d1": 1, "d3": 0}, "q1": {"d2": 2, "d1": 1}}
    assert ds.qrels["train"] == {"q1": {"d3": 1}}


def test_query_ids_keep_first_occurrence_order(tmp_path: Path) -> None:
    ds = load_beir(make_beir(tmp_path / "tiny"), name="custom")
    assert ds.name == "custom"
    assert query_ids(ds, "test") == ["q2", "q1"]
    assert query_ids(ds, "dev") == []


def test_dataset_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_beir(tmp_path / "tiny")
    assert dataset_dir("tiny", tmp_path) == tmp_path / "tiny"
    monkeypatch.setenv("FOOTNOTE_EVAL_DATA_DIR", str(tmp_path))
    assert dataset_dir("tiny") == tmp_path / "tiny"
    with pytest.raises(FileNotFoundError, match=r"scripts/fetch_eval_data\.sh"):
        dataset_dir("absent")
