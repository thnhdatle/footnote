# SPDX-License-Identifier: Apache-2.0
"""ONNX cross-encoder reranker."""

from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from footnote.adapters._onnx import (
    BatchEncoder,
    Session,
    load_tokenizer,
    open_session,
    run_model,
)
from footnote.adapters.models.manifest import ModelEntry, local_path, reranker_variant_key
from footnote.domain.errors import UpstreamError, ValidationError


class OnnxReranker:
    """Scores query and passage pairs with an ONNX cross-encoder."""

    def __init__(  # noqa: PLR0913 - session, tokenizer and the three file-derived values
        self,
        session: Session,
        tokenizer: Tokenizer,
        *,
        model_key: str,
        calibrated_threshold: float | None,
        max_tokens: int = 256,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValidationError("rerank.batch_size", "must be >= 1")
        self._session = session
        self._encoder = BatchEncoder(tokenizer, max_tokens)
        self._model_key = model_key
        self._threshold = calibrated_threshold
        self._batch_size = batch_size

    @classmethod
    def from_manifest(  # noqa: PLR0913 - mirrors the rerank.* settings
        cls,
        models_dir: Path,
        entry: ModelEntry,
        *,
        variant: str = "int8",
        arch: str | None = None,
        max_tokens: int = 256,
        batch_size: int = 32,
        intra_op_threads: int = 0,
    ) -> OnnxReranker:
        """Loads the tokenizer and model of a manifest entry."""
        file = entry.variants[reranker_variant_key(variant, arch)]
        return cls(
            open_session(local_path(models_dir, entry, file), intra_op_threads),
            load_tokenizer(local_path(models_dir, entry, entry.files["tokenizer"])),
            model_key=file.sha256,
            calibrated_threshold=file.calibrated_threshold,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def calibrated_threshold(self) -> float | None:
        return self._threshold

    def _score_all(self, query: str, passages: Sequence[str]) -> list[float]:
        out: list[float] = []
        for start in range(0, len(passages), self._batch_size):
            pairs = [(query, p) for p in passages[start : start + self._batch_size]]
            logits = np.asarray(run_model(self._session, self._encoder.encode(pairs)))
            flat = logits.astype(np.float64).reshape(len(pairs), -1)[:, 0]
            out.extend(float(x) for x in flat)
        return out

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Returns one raw logit per passage."""
        if not passages:
            return []
        try:
            return await asyncio.to_thread(self._score_all, query, list(passages))
        except (ValueError, IndexError) as exc:
            raise UpstreamError(f"reranker output has an unexpected shape: {exc}") from exc


class LazyOnnxReranker:
    """Reranker that loads the ONNX files on first use."""

    def __init__(  # noqa: PLR0913 - mirrors OnnxReranker.from_manifest
        self,
        models_dir: Path,
        entry: ModelEntry,
        *,
        variant: str = "int8",
        arch: str | None = None,
        max_tokens: int = 256,
        batch_size: int = 32,
        intra_op_threads: int = 0,
    ) -> None:
        self._file = entry.variants[reranker_variant_key(variant, arch)]
        self._load = functools.partial(
            OnnxReranker.from_manifest,
            models_dir,
            entry,
            variant=variant,
            arch=arch,
            max_tokens=max_tokens,
            batch_size=batch_size,
            intra_op_threads=intra_op_threads,
        )
        self._loaded: OnnxReranker | None = None
        self._lock = threading.Lock()

    @property
    def model_key(self) -> str:
        return self._file.sha256

    @property
    def calibrated_threshold(self) -> float | None:
        return self._file.calibrated_threshold

    def _get(self) -> OnnxReranker:
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load()
            return self._loaded

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        reranker = await asyncio.to_thread(self._get)
        return await reranker.score(query, passages)
