# SPDX-License-Identifier: Apache-2.0
"""Collection settings and their validation bounds."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import orjson

from footnote.domain.errors import ValidationError
from footnote.domain.webhooks import WebhookSettings

K_MAX = 100
LEG_K_MAX = 1000
RRF_K_MAX = 1000
RRF_WEIGHT_MAX = 10.0
RERANK_N_MAX = 200
CHUNK_MAX_TOKENS_MIN = 32
CHUNK_MAX_TOKENS_MAX = 2048


class ChunkingStrategyName(StrEnum):
    """Available chunking strategies."""

    HEADING = "heading"
    RECURSIVE = "recursive"
    FIXED = "fixed"


class LexicalMatch(StrEnum):
    """How the lexical leg turns the query into a tsquery."""

    ANY = "any"
    ALL = "all"
    WEBSEARCH = "websearch"


class LexicalRank(StrEnum):
    """Postgres ranking function for the lexical leg."""

    TS_RANK = "ts_rank"
    TS_RANK_CD = "ts_rank_cd"


class Language(StrEnum):
    """Supported FTS configurations."""

    SIMPLE = "simple"
    ENGLISH = "english"


class SearchMode(StrEnum):
    """Which retrieval legs a search runs."""

    VECTOR = "vector"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


def _check_int(field_name: str, value: int, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValidationError(field_name, f"must be an integer in [{low}, {high}]")


def _check_threshold(field_name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError(field_name, "must be in [0, 1] or null")


@dataclass(frozen=True, slots=True)
class ChunkingSettings:
    strategy: ChunkingStrategyName = ChunkingStrategyName.HEADING
    max_tokens: int = 400
    overlap_tokens: int = 40
    min_tokens: int = 50
    tokenizer: str = "bge-small-en-v1.5"

    def validate(self, tokenizer_ids: Collection[str] | None = None) -> None:
        _check_int(
            "chunking.max_tokens", self.max_tokens, CHUNK_MAX_TOKENS_MIN, CHUNK_MAX_TOKENS_MAX
        )
        _check_int("chunking.overlap_tokens", self.overlap_tokens, 0, self.max_tokens // 2)
        _check_int("chunking.min_tokens", self.min_tokens, 0, self.max_tokens)
        if tokenizer_ids is not None and self.tokenizer not in tokenizer_ids:
            raise ValidationError(
                "chunking.tokenizer", "must be a manifest id with a tokenizer file"
            )

    def settings_hash(self) -> str:
        payload = orjson.dumps(asdict(self), option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    lexical_match: LexicalMatch = LexicalMatch.ANY
    lexical_rank: LexicalRank = LexicalRank.TS_RANK
    k_lexical: int = 50
    k_vector: int = 50
    rrf_k: int = 60
    rrf_weights: dict[str, float] = field(default_factory=lambda: {"lexical": 0.2, "vector": 1.0})
    rerank: bool = True
    rerank_n: int = 30
    k: int = 8

    def validate(self) -> None:
        _check_int("retrieval.k", self.k, 1, K_MAX)
        _check_int("retrieval.k_lexical", self.k_lexical, 1, LEG_K_MAX)
        _check_int("retrieval.k_vector", self.k_vector, 1, LEG_K_MAX)
        _check_int("retrieval.rrf_k", self.rrf_k, 1, RRF_K_MAX)
        _check_int("retrieval.rerank_n", self.rerank_n, 1, RERANK_N_MAX)
        if self.rerank_n < self.k:
            raise ValidationError("retrieval.rerank_n", "must be >= k")
        if set(self.rrf_weights) != {"lexical", "vector"}:
            raise ValidationError("retrieval.rrf_weights", "must have exactly lexical and vector")
        for leg, weight in self.rrf_weights.items():
            if isinstance(weight, bool) or not math.isfinite(weight):
                raise ValidationError(f"retrieval.rrf_weights.{leg}", "must be a number in [0, 10]")
            if not 0.0 <= weight <= RRF_WEIGHT_MAX:
                raise ValidationError(f"retrieval.rrf_weights.{leg}", "must be in [0, 10]")
        if all(w == 0.0 for w in self.rrf_weights.values()):
            raise ValidationError("retrieval.rrf_weights", "must not both be 0")


@dataclass(frozen=True, slots=True)
class AnswerSettings:
    evidence_threshold_rerank: float | None = None
    evidence_threshold_vector: float | None = None
    prompt_template: str = "answer_v1"
    unsupported_as_abstain: bool = False

    def validate(self, templates: Collection[str] | None = None) -> None:
        _check_threshold("answer.evidence_threshold_rerank", self.evidence_threshold_rerank)
        _check_threshold("answer.evidence_threshold_vector", self.evidence_threshold_vector)
        if templates is not None and self.prompt_template not in templates:
            raise ValidationError("answer.prompt_template", "unknown template")


@dataclass(frozen=True, slots=True)
class CollectionSettings:
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    answer: AnswerSettings = field(default_factory=AnswerSettings)
    language: Language = Language.SIMPLE
    embedding_space: str = "bge-small-en-v1.5-onnx"
    webhook: WebhookSettings = field(default_factory=WebhookSettings)
