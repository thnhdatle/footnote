# SPDX-License-Identifier: Apache-2.0
"""Conversions between domain values and their stored form."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import orjson
import zstandard

from footnote.domain.documents import Element
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.settings import (
    AnswerSettings,
    ChunkingSettings,
    ChunkingStrategyName,
    CollectionSettings,
    Language,
    LexicalMatch,
    LexicalRank,
    RetrievalSettings,
)
from footnote.domain.tenancy import Quotas
from footnote.domain.webhooks import WebhookEvent, WebhookSettings

_ZSTD_LEVEL = 3


def compress(data: bytes) -> bytes:
    """zstd-compress stored originals and element lists."""
    return zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data)


def decompress(data: bytes) -> bytes:
    """Inverse of `compress` (frames carry their content size)."""
    return zstandard.ZstdDecompressor().decompress(data)


def encode_elements(elements: Sequence[Element]) -> bytes:
    """Elements as zstd-compressed JSON."""
    return compress(orjson.dumps([e.to_dict() for e in elements]))


def decode_elements(data: bytes) -> list[Element]:
    """Inverse of `encode_elements`."""
    return [Element.from_dict(item) for item in orjson.loads(decompress(data))]


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text form `[x,y,…]`, cast in SQL to `vector(<d>)`/`halfvec(<d>)`."""
    values = [float(x) for x in vector]
    if not values or not all(math.isfinite(x) for x in values):
        msg = "vector must be non-empty and finite"
        raise ValueError(msg)
    return "[" + ",".join(repr(x) for x in values) + "]"


def settings_to_json(settings: CollectionSettings) -> dict[str, Any]:
    """Stored part of the settings; `language` and the space live in their own columns."""
    return {
        "chunking": asdict(settings.chunking),
        "retrieval": asdict(settings.retrieval),
        "answer": asdict(settings.answer),
        "webhook": {"events": [e.value for e in settings.webhook.events]},
    }


def settings_from_json(
    data: Mapping[str, Any], language: str, embedding_space: str
) -> CollectionSettings:
    """Rebuild CollectionSettings; missing keys take the domain defaults."""
    chunking = {**asdict(ChunkingSettings()), **dict(data.get("chunking") or {})}
    retrieval = {**asdict(RetrievalSettings()), **dict(data.get("retrieval") or {})}
    answer = {**asdict(AnswerSettings()), **dict(data.get("answer") or {})}
    hook = data.get("webhook") or {}
    events = hook.get("events")
    webhook = (
        WebhookSettings()
        if events is None
        else WebhookSettings(events=tuple(WebhookEvent(e) for e in events))
    )
    return CollectionSettings(
        chunking=ChunkingSettings(
            strategy=ChunkingStrategyName(chunking["strategy"]),
            max_tokens=int(chunking["max_tokens"]),
            overlap_tokens=int(chunking["overlap_tokens"]),
            min_tokens=int(chunking["min_tokens"]),
            tokenizer=str(chunking["tokenizer"]),
        ),
        retrieval=RetrievalSettings(
            lexical_match=LexicalMatch(retrieval["lexical_match"]),
            lexical_rank=LexicalRank(retrieval["lexical_rank"]),
            k_lexical=int(retrieval["k_lexical"]),
            k_vector=int(retrieval["k_vector"]),
            rrf_k=int(retrieval["rrf_k"]),
            rrf_weights={str(k): float(v) for k, v in retrieval["rrf_weights"].items()},
            rerank=bool(retrieval["rerank"]),
            rerank_n=int(retrieval["rerank_n"]),
            k=int(retrieval["k"]),
        ),
        answer=AnswerSettings(
            evidence_threshold_rerank=_opt_float(answer["evidence_threshold_rerank"]),
            evidence_threshold_vector=_opt_float(answer["evidence_threshold_vector"]),
            prompt_template=str(answer["prompt_template"]),
            unsupported_as_abstain=bool(answer["unsupported_as_abstain"]),
        ),
        language=Language(language),
        embedding_space=embedding_space,
        webhook=webhook,
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def quotas_to_json(quotas: Quotas) -> dict[str, int]:
    """Quotas as stored in `tenants.quotas`."""
    return asdict(quotas)


def quotas_from_json(data: Mapping[str, Any]) -> Quotas:
    """Inverse of `quotas_to_json`; missing limits are 0 (unlimited)."""
    base = asdict(Quotas())
    return Quotas(**{k: int(data.get(k, v)) for k, v in base.items()})


def space_from_row(row: Mapping[str, Any]) -> EmbeddingSpace:
    """EmbeddingSpace from an `embedding_spaces` row."""
    threshold = row["calibrated_threshold"]
    return EmbeddingSpace(
        id=int(row["id"]),
        name=str(row["name"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        dimensions=int(row["dimensions"]),
        send_dimensions=bool(row["send_dimensions"]),
        normalise=bool(row["normalise"]),
        calibrated_threshold=None if threshold is None else float(threshold),
    )
