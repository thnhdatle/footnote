# SPDX-License-Identifier: Apache-2.0
"""In-process `bm25s` baseline: BM25 over the BEIR corpus, documents ranked
directly (no chunking), English stopwords and the Snowball English stemmer (PyStemmer).

Needs the `eval` extra (`bm25s`, `PyStemmer`); `MissingExtraError` names it otherwise.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from footnote.app.evaluation.datasets import CorpusDoc

EXTRA_HINT = "install the eval extra: uv sync --extra eval (bm25s, PyStemmer)"


class MissingExtraError(RuntimeError):
    """The `eval` optional dependencies are not installed."""


def _modules() -> tuple[Any, Any]:
    try:
        import bm25s  # noqa: PLC0415 - optional extra
        import Stemmer  # noqa: PLC0415 - optional extra (PyStemmer)
    except ImportError as exc:
        raise MissingExtraError(f"bm25s baseline unavailable: {EXTRA_HINT}") from exc
    return bm25s, Stemmer


def require_extra() -> None:
    """Raise MissingExtraError unless `bm25s` and `PyStemmer` import."""
    _modules()


def document_text(doc: CorpusDoc) -> str:
    """Indexed text of a corpus document: title and text."""
    return f"{doc.title}\n{doc.text}" if doc.title else doc.text


@dataclass(frozen=True)
class Bm25Run:
    """Ranked document ids and the in-process retrieval latency (seconds) per query."""

    rankings: dict[str, list[str]]
    seconds: dict[str, float]


class Bm25Index:
    """A `bm25s` (Lucene BM25, k1 1.5, b 0.75) index over a corpus."""

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
        """Document ids with a positive score, best first, at most `k`."""
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
        """Rank every query (one at a time, so each latency is a single query's)."""
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
    """Weighted RRF of document rankings (`(ranking, weight)` per leg). Ties break on the
    position in `tiebreak` (documents absent from it last), then document id."""
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
