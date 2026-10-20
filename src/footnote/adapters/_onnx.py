# SPDX-License-Identifier: Apache-2.0
"""Shared ONNX Runtime and tokenizer plumbing of the local embedder and reranker."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from footnote.domain.errors import UnavailableError

Int64Array = npt.NDArray[np.int64]


class Session(Protocol):
    """The part of `onnxruntime.InferenceSession` the adapters use."""

    def get_inputs(self) -> Sequence[Any]:
        """Model inputs (objects with a `name`)."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        """Run inference."""


def open_session(path: Path, intra_op_threads: int = 0) -> Session:
    """CPU inference session; `intra_op_threads` 0 keeps the ONNX Runtime default."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    try:
        session: Session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # onnxruntime raises several unrelated types
        raise UnavailableError(f"cannot load onnx model {path}: {exc}") from exc
    return session


def load_tokenizer(path: Path) -> Tokenizer:
    """Load `tokenizer.json`; UnavailableError when it cannot be read."""
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:  # tokenizers raises a bare Exception
        raise UnavailableError(f"cannot load tokenizer {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Batch:
    """Padded model inputs of one batch and how many inputs were cut at the token limit."""

    input_ids: Int64Array
    attention_mask: Int64Array
    token_type_ids: Int64Array
    truncated: int


class BatchEncoder:
    """Tokenizes texts or text pairs with truncation at `max_tokens` and batch padding."""

    def __init__(self, tokenizer: Tokenizer, max_tokens: int) -> None:
        tokenizer.enable_truncation(max_length=max_tokens)
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # noqa: S106
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def encode(self, inputs: Sequence[str] | Sequence[tuple[str, str]]) -> Batch:
        """Encode one batch; the tokenizer is shared, so encoding is serialised."""
        with self._lock:
            encodings = self._tokenizer.encode_batch(list(inputs))
        return Batch(
            input_ids=np.array([e.ids for e in encodings], dtype=np.int64),
            attention_mask=np.array([e.attention_mask for e in encodings], dtype=np.int64),
            token_type_ids=np.array([e.type_ids for e in encodings], dtype=np.int64),
            truncated=sum(1 for e in encodings if e.overflowing),
        )


def run_model(session: Session, batch: Batch) -> Any:
    """Run `session` feeding only the inputs the model declares; returns the first output."""
    available = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "token_type_ids": batch.token_type_ids,
    }
    feed = {i.name: available[i.name] for i in session.get_inputs() if i.name in available}
    return session.run(None, feed)[0]
