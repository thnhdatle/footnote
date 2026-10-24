# SPDX-License-Identifier: Apache-2.0
"""Token counter over a Hugging Face tokenizer file."""

from __future__ import annotations

import threading
from pathlib import Path

from tokenizers import Tokenizer

from footnote.domain.errors import UnavailableError


class HfTokenCounter:
    """Counts tokens with a Hugging Face tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: Path) -> HfTokenCounter:
        """Loads a tokenizer.json file."""
        try:
            return cls(Tokenizer.from_file(str(path)))
        except Exception as exc:
            raise UnavailableError(f"cannot load tokenizer {path}: {exc}") from exc

    def count(self, text: str) -> int:
        """Counts the tokens of a text."""
        with self._lock:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
