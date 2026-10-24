# SPDX-License-Identifier: Apache-2.0
"""Shared ONNX Runtime and tokenizer helpers of the local embedder and reranker."""

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
    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]: ...


def open_session(path: Path, intra_op_threads: int = 0) -> Session:
    """Opens a CPU inference session."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    try:
        session: Session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise UnavailableError(f"cannot load onnx model {path}: {exc}") from exc
    return session


def load_tokenizer(path: Path) -> Tokenizer:
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:
        raise UnavailableError(f"cannot load tokenizer {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Batch:
    input_ids: Int64Array
    attention_mask: Int64Array
    token_type_ids: Int64Array
    truncated: int


class BatchEncoder:
    """Tokenizes texts or text pairs into padded batches, truncated at a token limit."""

    def __init__(self, tokenizer: Tokenizer, max_tokens: int) -> None:
        tokenizer.enable_truncation(max_length=max_tokens)
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # noqa: S106
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def encode(self, inputs: Sequence[str] | Sequence[tuple[str, str]]) -> Batch:
        with self._lock:
            encodings = self._tokenizer.encode_batch(list(inputs))
        return Batch(
            input_ids=np.array([e.ids for e in encodings], dtype=np.int64),
            attention_mask=np.array([e.attention_mask for e in encodings], dtype=np.int64),
            token_type_ids=np.array([e.type_ids for e in encodings], dtype=np.int64),
            truncated=sum(1 for e in encodings if e.overflowing),
        )


def run_model(session: Session, batch: Batch) -> Any:
    """Runs a session on one encoded batch."""
    available = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "token_type_ids": batch.token_type_ids,
    }
    feed = {i.name: available[i.name] for i in session.get_inputs() if i.name in available}
    return session.run(None, feed)[0]
