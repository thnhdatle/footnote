# SPDX-License-Identifier: Apache-2.0
"""Local ONNX embedder (manifest model on CPU, batched, CLS or mean pooling)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
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
from footnote.adapters.models.manifest import ModelEntry, local_path
from footnote.domain.errors import ValidationError

DEFAULT_BATCH_SIZE = 32


class OnnxEmbedder:
    """Implements `ports.Embedder` for one local embedding model."""

    def __init__(  # noqa: PLR0913 - one value per manifest field plus the two engines
        self,
        session: Session,
        tokenizer: Tokenizer,
        *,
        max_tokens: int,
        pooling: str = "cls",
        query_prefix: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_truncated: Callable[[int], None] | None = None,
    ) -> None:
        if pooling not in {"cls", "mean"}:
            raise ValidationError("pooling", f"{pooling!r} must be cls or mean")
        if batch_size < 1:
            raise ValidationError("batch_size", "must be >= 1")
        self._session = session
        self._encoder = BatchEncoder(tokenizer, max_tokens)
        self._pooling = pooling
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._on_truncated = on_truncated

    @classmethod
    def from_manifest(
        cls,
        models_dir: Path,
        entry: ModelEntry,
        *,
        intra_op_threads: int = 0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_truncated: Callable[[int], None] | None = None,
    ) -> OnnxEmbedder:
        """Load the model and tokenizer files of a manifest entry from `models_dir`."""
        model = local_path(models_dir, entry, entry.files["model"])
        tokenizer = local_path(models_dir, entry, entry.files["tokenizer"])
        return cls(
            open_session(model, intra_op_threads),
            load_tokenizer(tokenizer),
            max_tokens=entry.max_tokens,
            pooling=entry.pooling,
            query_prefix=entry.query_prefix,
            batch_size=batch_size,
            on_truncated=on_truncated,
        )

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        batch = self._encoder.encode(texts)
        if batch.truncated and self._on_truncated is not None:
            self._on_truncated(batch.truncated)
        hidden = np.asarray(run_model(self._session, batch), dtype=np.float32)
        if self._pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = batch.attention_mask[:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        unit = pooled / np.where(norms == 0.0, 1.0, norms)
        return [[float(x) for x in row] for row in unit]

    def _embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            out.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts, one L2-normalised vector per input."""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_all, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query with the manifest prefix prepended."""
        vectors = await asyncio.to_thread(self._embed_all, [self._query_prefix + text])
        return vectors[0]
