# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from footnote.domain.errors import ValidationError
from footnote.domain.settings import (
    AnswerSettings,
    ChunkingSettings,
    ChunkingStrategyName,
    CollectionSettings,
    Language,
    RetrievalSettings,
)

NAN = float("nan")
INF = float("inf")


def _retrieval(**kw: Any) -> RetrievalSettings:
    return replace(RetrievalSettings(), **kw)


def _bad_field(settings: RetrievalSettings | ChunkingSettings | AnswerSettings) -> str:
    with pytest.raises(ValidationError) as info:
        settings.validate()
    return info.value.field


def test_defaults_are_valid() -> None:
    RetrievalSettings().validate()
    ChunkingSettings().validate({"bge-small-en-v1.5"})
    AnswerSettings().validate({"answer_v1"})
    collection = CollectionSettings()
    assert collection.language is Language.SIMPLE
    assert collection.chunking.strategy is ChunkingStrategyName.HEADING


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [("k_lexical", 1, 1000), ("k_vector", 1, 1000), ("rrf_k", 1, 1000)],
)
def test_retrieval_int_bounds(name: str, low: int, high: int) -> None:
    _retrieval(**{name: low}).validate()
    _retrieval(**{name: high}).validate()
    for bad in (low - 1, high + 1):
        assert _bad_field(_retrieval(**{name: bad})) == f"retrieval.{name}"


def test_retrieval_k_bounds() -> None:
    _retrieval(k=1).validate()
    _retrieval(k=100, rerank_n=100).validate()
    assert _bad_field(_retrieval(k=0)) == "retrieval.k"
    assert _bad_field(_retrieval(k=101, rerank_n=200)) == "retrieval.k"


def test_retrieval_rerank_n_bounds_and_relation_to_k() -> None:
    _retrieval(rerank_n=1, k=1).validate()
    _retrieval(rerank_n=200).validate()
    assert _bad_field(_retrieval(rerank_n=0)) == "retrieval.rerank_n"
    assert _bad_field(_retrieval(rerank_n=201)) == "retrieval.rerank_n"
    _retrieval(rerank_n=8, k=8).validate()
    assert _bad_field(_retrieval(rerank_n=7, k=8)) == "retrieval.rerank_n"


@pytest.mark.parametrize("bad", [True, 1.5, "5", None])
def test_retrieval_rejects_non_integers(bad: Any) -> None:
    assert _bad_field(_retrieval(k_vector=bad)) == "retrieval.k_vector"


@pytest.mark.parametrize("leg", ["lexical", "vector"])
def test_rrf_weight_bounds(leg: str) -> None:
    other = "vector" if leg == "lexical" else "lexical"
    for ok in (0.0, 10.0, 3):
        _retrieval(rrf_weights={leg: ok, other: 1.0}).validate()
    for bad in (-0.001, 10.001):
        field = _bad_field(_retrieval(rrf_weights={leg: bad, other: 1.0}))
        assert field == f"retrieval.rrf_weights.{leg}"


@pytest.mark.parametrize("bad", [NAN, INF, -INF, True])
def test_rrf_weight_rejects_non_numbers(bad: Any) -> None:
    field = _bad_field(_retrieval(rrf_weights={"lexical": bad, "vector": 1.0}))
    assert field == "retrieval.rrf_weights.lexical"


def test_rrf_weights_not_both_zero() -> None:
    _retrieval(rrf_weights={"lexical": 0.0, "vector": 1.0}).validate()
    _retrieval(rrf_weights={"lexical": 1.0, "vector": 0.0}).validate()
    assert _bad_field(_retrieval(rrf_weights={"lexical": 0, "vector": 0})) == (
        "retrieval.rrf_weights"
    )


@pytest.mark.parametrize(
    "weights",
    [{}, {"lexical": 1.0}, {"vector": 1.0}, {"lexical": 1, "vector": 1, "x": 1}, {"a": 1, "b": 1}],
)
def test_rrf_weights_exact_keys(weights: dict[str, float]) -> None:
    assert _bad_field(_retrieval(rrf_weights=weights)) == "retrieval.rrf_weights"


def test_chunking_max_tokens_bounds() -> None:
    ChunkingSettings(max_tokens=32, overlap_tokens=0, min_tokens=0).validate()
    ChunkingSettings(max_tokens=2048).validate()
    assert _bad_field(ChunkingSettings(max_tokens=31)) == "chunking.max_tokens"
    assert _bad_field(ChunkingSettings(max_tokens=2049)) == "chunking.max_tokens"


def test_chunking_overlap_bounds() -> None:
    ChunkingSettings(max_tokens=100, overlap_tokens=0).validate()
    ChunkingSettings(max_tokens=100, overlap_tokens=50).validate()
    ChunkingSettings(max_tokens=101, overlap_tokens=50).validate()
    assert _bad_field(ChunkingSettings(max_tokens=100, overlap_tokens=51)) == (
        "chunking.overlap_tokens"
    )
    assert _bad_field(ChunkingSettings(max_tokens=100, overlap_tokens=-1)) == (
        "chunking.overlap_tokens"
    )


def test_chunking_min_tokens_bounds() -> None:
    ChunkingSettings(max_tokens=100, min_tokens=0).validate()
    ChunkingSettings(max_tokens=100, min_tokens=100).validate()
    assert _bad_field(ChunkingSettings(max_tokens=100, min_tokens=101)) == "chunking.min_tokens"
    assert _bad_field(ChunkingSettings(max_tokens=100, min_tokens=-1)) == "chunking.min_tokens"


def test_chunking_tokenizer_must_be_in_ids() -> None:
    ChunkingSettings(tokenizer="a").validate({"a", "b"})
    with pytest.raises(ValidationError) as info:
        ChunkingSettings(tokenizer="c").validate({"a", "b"})
    assert info.value.field == "chunking.tokenizer"
    ChunkingSettings(tokenizer="anything").validate(None)


def test_settings_hash_stable_and_sensitive() -> None:
    base = ChunkingSettings()
    assert base.settings_hash() == ChunkingSettings().settings_hash()
    assert len(base.settings_hash()) == 64
    variants = [
        replace(base, strategy=ChunkingStrategyName.FIXED),
        replace(base, max_tokens=401),
        replace(base, overlap_tokens=41),
        replace(base, min_tokens=51),
        replace(base, tokenizer="other"),
    ]
    hashes = {v.settings_hash() for v in variants} | {base.settings_hash()}
    assert len(hashes) == len(variants) + 1


@pytest.mark.parametrize("name", ["evidence_threshold_rerank", "evidence_threshold_vector"])
def test_answer_threshold_bounds(name: str) -> None:
    for ok in (None, 0.0, 1.0, 0.5, 0, 1):
        AnswerSettings(**{name: ok}).validate()
    for bad in (-0.01, 1.01, NAN, INF, -INF, True, False):
        assert _bad_field(AnswerSettings(**{name: bad})) == f"answer.{name}"


def test_answer_template_must_be_known() -> None:
    AnswerSettings(prompt_template="answer_v1").validate({"answer_v1"})
    with pytest.raises(ValidationError) as info:
        AnswerSettings(prompt_template="nope").validate({"answer_v1"})
    assert info.value.field == "answer.prompt_template"
