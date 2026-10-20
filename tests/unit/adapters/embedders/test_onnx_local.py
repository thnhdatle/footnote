# SPDX-License-Identifier: Apache-2.0
"""OnnxEmbedder with a fake session: pooling, normalisation, batching, truncation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from footnote.adapters.embedders.onnx_local import OnnxEmbedder
from footnote.domain.errors import ValidationError
from tests.unit.adapters.conftest import VOCAB, FakeSession


def _tok(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def _unit(v: np.ndarray) -> list[float]:
    return [float(x) for x in v / np.linalg.norm(v)]


async def test_cls_pooling_is_normalised_first_token(tokenizer_file: Path) -> None:
    session = FakeSession()
    emb = OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16)
    [vec] = await emb.embed_documents(["a b"])
    assert vec == pytest.approx(_unit(session.table[VOCAB["[CLS]"]]), abs=1e-6)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


async def test_mean_pooling_ignores_padding(tokenizer_file: Path) -> None:
    session = FakeSession()
    emb = OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16, pooling="mean")
    alone = await emb.embed_documents(["a"])
    batched = await emb.embed_documents(["a", "a b c d"])
    ids = [VOCAB[t] for t in ("[CLS]", "a", "[SEP]")]
    assert alone[0] == pytest.approx(_unit(session.table[ids].mean(axis=0)), abs=1e-6)
    assert batched[0] == pytest.approx(alone[0], abs=1e-6)


async def test_query_prefix_is_prepended_to_queries_only(tokenizer_file: Path) -> None:
    session = FakeSession()
    emb = OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16, query_prefix="a ")
    await emb.embed_query("b")
    await emb.embed_documents(["b"])
    assert session.feeds[0]["input_ids"].tolist() == [[2, 4, 5, 3]]
    assert session.feeds[1]["input_ids"].tolist() == [[2, 5, 3]]


async def test_batches_and_order(tokenizer_file: Path) -> None:
    session = FakeSession()
    emb = OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16, batch_size=2, pooling="mean")
    texts = ["a", "b c", "d", "e f g", "h"]
    out = await emb.embed_documents(texts)
    assert len(session.feeds) == 3
    singles = [(await emb.embed_documents([t]))[0] for t in texts]
    for got, want in zip(out, singles, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


async def test_feeds_only_declared_inputs(tokenizer_file: Path) -> None:
    session = FakeSession(("input_ids", "attention_mask", "token_type_ids"))
    await OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16).embed_documents(["a"])
    assert sorted(session.feeds[0]) == ["attention_mask", "input_ids", "token_type_ids"]
    session = FakeSession(("input_ids",))
    await OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=16).embed_documents(["a"])
    assert sorted(session.feeds[0]) == ["input_ids"]


async def test_truncation_is_counted(tokenizer_file: Path) -> None:
    seen: list[int] = []
    session = FakeSession()
    emb = OnnxEmbedder(session, _tok(tokenizer_file), max_tokens=4, on_truncated=seen.append)
    await emb.embed_documents(["a b c d e f", "a"])
    assert seen == [1]
    assert session.feeds[0]["input_ids"].shape == (2, 4)


async def test_empty_documents(tokenizer_file: Path) -> None:
    emb = OnnxEmbedder(FakeSession(), _tok(tokenizer_file), max_tokens=4)
    assert await emb.embed_documents([]) == []


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("pooling", {"pooling": "max"}), ("batch", {"batch_size": 0})],
)
def test_invalid_options(tokenizer_file: Path, name: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OnnxEmbedder(FakeSession(), _tok(tokenizer_file), max_tokens=4, **kwargs)  # type: ignore[arg-type]
