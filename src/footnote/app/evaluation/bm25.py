# SPDX-License-Identifier: Apache-2.0
"""In-process BM25 baseline over the BEIR corpus."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from footnote.app.evaluation.datasets import CorpusDoc

EXTRA_HINT = "install the eval extra: uv sync --extra eval (bm25s, PyStemmer)"


class MissingExtraError(RuntimeError): ...


def _modules() -> tuple[Any, Any]:
    try:
        import bm25s  # noqa: PLC0415 - optional extra
        import Stemmer  # noqa: PLC0415 - optional extra (PyStemmer)
    except ImportError as exc:
        raise MissingExtraError(f"bm25s baseline unavailable: {EXTRA_HINT}") from exc
    return bm25s, Stemmer


def require_extra() -> None:
    _modules()


def document_text(doc: CorpusDoc) -> str:
    return f"{doc.title}\n{doc.text}" if doc.title else doc.text


@dataclass(frozen=True)
class Bm25Run:
    rankings: dict[str, list[str]]
    seconds: dict[str, float]


class Bm25Index:
    """A BM25 index over a corpus."""

    def __init__(self, corpus: Mapping[str, CorpusDoc]) -> None:
        bm25s, stemmer_mod = _modules()
        self._bm25s = bm25s
        self._stemmer = stemmer_mod.Stemmer("english")
        self._ids = list(corpus)
        tokens = bm25s.tokenize(
            [document_text(corpus[i]) for i in self._ids],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 100) -> list[str]:
        tokens = self._bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self._stemmer,
            return_ids=False,
            show_progress=False,
        )
        if not tokens[0]:
            return []
        docs, scores = self._retriever.retrieve(
            tokens, k=min(k, len(self._ids)), show_progress=False, n_threads=1
        )
        return [
            self._ids[int(d)] for d, s in zip(docs[0], scores[0], strict=True) if float(s) > 0.0
        ]

    def run(
        self,
        queries: Mapping[str, str],
        k: int = 100,
        clock: Callable[[], float] = time.perf_counter,
    ) -> Bm25Run:
        rankings: dict[str, list[str]] = {}
        seconds: dict[str, float] = {}
        for qid, text in queries.items():
            start = clock()
            rankings[qid] = self.search(text, k)
            seconds[qid] = clock() - start
        return Bm25Run(rankings, seconds)


def fuse_rankings(
    legs: Sequence[tuple[Sequence[str], float]], rrf_k: int = 60, tiebreak: Sequence[str] = ()
) -> list[str]:
    totals: dict[str, float] = {}
    for ranking, weight in legs:
        seen: set[str] = set()
        for position, doc in enumerate(ranking, start=1):
            if doc in seen:
                continue
            seen.add(doc)
            totals[doc] = totals.get(doc, 0.0) + weight / (rrf_k + position)
    order = {doc: i for i, doc in enumerate(dict.fromkeys(tiebreak))}
    missing = len(order)
    return sorted(totals, key=lambda d: (-totals[d], order.get(d, missing), d))
