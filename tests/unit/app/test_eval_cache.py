# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from footnote.app.evaluation.cache import (
    CachingEmbedder,
    cache_key_for_space,
    cache_path,
    is_warm,
    mark_warm,
)
from footnote.domain.embedding import EmbeddingSpace


class Inner:
    def __init__(self) -> None:
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def vec(text: str) -> list[float]:
        return [len(text) / 3.0, 0.1, 1e-9 * len(text)]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [self.vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [-x for x in self.vec(text)]


async def test_only_misses_reach_the_inner_embedder_in_order(tmp_path: Path) -> None:
    inner = Inner()
    cache = CachingEmbedder(inner, tmp_path, "k")
    first = await cache.embed_documents(["aa", "b", "aa"])
    assert inner.doc_calls == [["aa", "b"]]
    second = await cache.embed_documents(["c", "b", "aa"])
    assert inner.doc_calls == [["aa", "b"], ["c"]]
    assert second[1] == first[1]
    assert second[2] == first[0] == first[2]
    assert await cache.embed_documents(["b", "aa"]) == [first[1], first[0]]
    assert len(inner.doc_calls) == 2
    cache.close()


async def test_cached_and_uncached_results_are_identical(tmp_path: Path) -> None:
    texts = ["alpha", "be", "gamma delta"]
    cold = CachingEmbedder(Inner(), tmp_path, "k")
    miss = await cold.embed_documents(texts)
    cold.close()
    inner = Inner()
    warm = CachingEmbedder(inner, tmp_path, "k")
    hit = await warm.embed_documents(texts)
    assert inner.doc_calls == []
    assert hit == miss
    assert miss[0] != Inner.vec(texts[0])
    warm.close()


async def test_documents_and_queries_are_separate_kinds(tmp_path: Path) -> None:
    inner = Inner()
    cache = CachingEmbedder(inner, tmp_path, "k")
    doc = await cache.embed_documents(["same"])
    query = await cache.embed_query("same")
    assert query == pytest.approx([-x for x in doc[0]])
    assert await cache.embed_query("same") == query
    assert inner.query_calls == ["same"]
    assert len(inner.doc_calls) == 1
    cache.close()


async def test_one_file_per_key_with_meta(tmp_path: Path) -> None:
    a = CachingEmbedder(Inner(), tmp_path, "key-a")
    b = CachingEmbedder(Inner(), tmp_path, "key-b")
    assert a.path != b.path
    assert a.path == tmp_path / (hashlib.sha256(b"key-a").hexdigest()[:16] + ".sqlite")
    a.close()
    b.close()
    with sqlite3.connect(a.path) as db:
        assert db.execute("SELECT value FROM meta WHERE key = 'key'").fetchone() == ("key-a",)


def test_cache_keys() -> None:
    openai = EmbeddingSpace(1, "s", "openai_compat", "m", 8)
    assert cache_key_for_space(openai) == "openai_compat/m/8"
    onnx = EmbeddingSpace(2, "o", "onnx_local", "bge", 384)
    assert cache_key_for_space(onnx, "abc") == "onnx_local/abc"
    with pytest.raises(ValueError, match="sha256"):
        cache_key_for_space(onnx)


async def test_warm_markers(tmp_path: Path) -> None:
    assert not is_warm(tmp_path, "scifact")
    mark_warm(tmp_path, "scifact", "k", 5)
    assert not is_warm(tmp_path, "scifact")
    CachingEmbedder(Inner(), tmp_path, "k").close()
    assert is_warm(tmp_path, "scifact")
    assert cache_path(tmp_path, "k").is_file()
    (tmp_path / "warm" / "scifact.json").write_text("not json")
    assert not is_warm(tmp_path, "scifact")


async def test_queries_bypass_the_cache_when_disabled(tmp_path: Path) -> None:
    inner = Inner()
    cache = CachingEmbedder(inner, tmp_path, "k", cache_queries=False)
    first = await cache.embed_query("q")
    second = await cache.embed_query("q")
    assert inner.query_calls == ["q", "q"]
    assert first == second
    cached = CachingEmbedder(Inner(), tmp_path / "other", "k")
    assert await cached.embed_query("q") == first
    assert await cache.embed_documents(["d"]) == await cache.embed_documents(["d"])
    assert len(inner.doc_calls) == 1
    cache.close()
    cached.close()
