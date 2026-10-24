# SPDX-License-Identifier: Apache-2.0
"""Loads BEIR-layout evaluation datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import orjson

DATA_DIR_ENV = "FOOTNOTE_EVAL_DATA_DIR"
DEFAULT_DATA_DIR = "~/.cache/rayld/footnote/datasets"


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BeirDataset:
    name: str
    corpus: dict[str, CorpusDoc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, dict[str, int]]]


def dataset_dir(name: str, data_dir: Path | None = None) -> Path:
    base = data_dir or Path(os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR).expanduser()
    path = base / name
    if not (path / "corpus.jsonl").is_file():
        msg = f"dataset {name!r} not found at {path}; run scripts/fetch_eval_data.sh {name}"
        raise FileNotFoundError(msg)
    return path


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:  # noqa: PLR2004 - query, doc, score
                continue
            result.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return result


def load_beir(path: Path, name: str | None = None) -> BeirDataset:
    """Loads a dataset directory in BEIR layout."""
    corpus: dict[str, CorpusDoc] = {}
    for row in _jsonl(path / "corpus.jsonl"):
        doc_id = str(row["_id"])
        corpus[doc_id] = CorpusDoc(doc_id, str(row.get("title") or ""), str(row.get("text") or ""))
    queries = {str(row["_id"]): str(row["text"]) for row in _jsonl(path / "queries.jsonl")}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    qrels_dir = path / "qrels"
    if qrels_dir.is_dir():
        for file in sorted(qrels_dir.glob("*.tsv")):
            qrels[file.stem] = _load_qrels(file)
    return BeirDataset(name or path.name, corpus, queries, qrels)


def query_ids(dataset: BeirDataset, split: str) -> list[str]:
    return list(dataset.qrels.get(split, {}))
