# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

WORDS = ["a", "b", "c", "d", "e", "f", "g", "h"]
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: 4 + i for i, w in enumerate(WORDS)}}
HIDDEN = 4


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tok


@pytest.fixture
def tokenizer_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    build_tokenizer().save(str(path))
    return path


class FakeSession:
    def __init__(self, input_names: Sequence[str] = ("input_ids", "attention_mask")) -> None:
        rng = np.random.default_rng(7)
        self.table = rng.normal(size=(len(VOCAB), HIDDEN)).astype(np.float32)
        self.names = list(input_names)
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name=n) for n in self.names]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(input_feed)
        return [self.table[input_feed["input_ids"]]]


class LogitSession(FakeSession):
    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(input_feed)
        ids = input_feed["input_ids"]
        mask = input_feed["attention_mask"]
        logits = (self.table[ids][:, :, 0] * mask).sum(axis=1, keepdims=True)
        return [logits.astype(np.float32)]
