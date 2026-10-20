# SPDX-License-Identifier: Apache-2.0
"""`HfTokenCounter`: token counting with a Hugging Face `tokenizer.json` (`ports.TokenCounter`)."""

from __future__ import annotations

import threading
from pathlib import Path

from tokenizers import Tokenizer

from footnote.domain.errors import UnavailableError


class HfTokenCounter:
    """Counts tokens without special tokens, as the chunker's budgets expect."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: Path) -> HfTokenCounter:
        """Load `tokenizer.json`; UnavailableError when it cannot be read."""
        try:
            return cls(Tokenizer.from_file(str(path)))
        except Exception as exc:  # tokenizers raises a bare Exception
            raise UnavailableError(f"cannot load tokenizer {path}: {exc}") from exc

    def count(self, text: str) -> int:
        """Number of tokens in `text`, excluding special tokens."""
        with self._lock:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
