# SPDX-License-Identifier: Apache-2.0
"""OnnxReranker with a fake session, and the noop reranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer

import footnote.adapters.rerankers.onnx_cross_encoder as module
from footnote.adapters.models.manifest import ModelEntry, ModelFile
from footnote.adapters.rerankers import NoopReranker, OnnxReranker
from footnote.domain.errors import UpstreamError, ValidationError
from tests.unit.adapters.conftest import VOCAB, FakeSession, LogitSession


def _reranker(path: Path, session: FakeSession, **kw: Any) -> OnnxReranker:
    return OnnxReranker(
        session,
        Tokenizer.from_file(str(path)),
        model_key="k",
        calibrated_threshold=0.4,
        **kw,
    )


async def test_scores_pairs_in_order(tokenizer_file: Path) -> None:
    session = LogitSession()
    rr = _reranker(tokenizer_file, session, max_tokens=16)
    logits = await rr.score("a", ["b", "c d"])
    first = session.table[[VOCAB[t] for t in ("[CLS]", "a", "[SEP]", "b", "[SEP]")], 0].sum()
    second = session.table[[VOCAB[t] for t in ("[CLS]", "a", "[SEP]", "c", "d", "[SEP]")], 0].sum()
    assert logits == pytest.approx([first, second], abs=1e-5)
    assert rr.model_key == "k"
    assert rr.calibrated_threshold == 0.4


async def test_batching_matches_single_scores(tokenizer_file: Path) -> None:
    session = LogitSession()
    rr = _reranker(tokenizer_file, session, max_tokens=16, batch_size=2)
    passages = ["b", "c d", "e", "f g h", "a"]
    out = await rr.score("a b", passages)
    assert len(session.feeds) == 3
    singles = [(await rr.score("a b", [p]))[0] for p in passages]
    assert out == pytest.approx(singles, abs=1e-5)


async def test_pair_truncation(tokenizer_file: Path) -> None:
    session = LogitSession()
    await _reranker(tokenizer_file, session, max_tokens=6).score("a b c", ["d e f g h"])
    assert session.feeds[0]["input_ids"].shape == (1, 6)


async def test_empty_passages(tokenizer_file: Path) -> None:
    assert await _reranker(tokenizer_file, LogitSession()).score("q", []) == []


async def test_bad_output_shape(tokenizer_file: Path) -> None:
    class Bad(FakeSession):
        def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
            return [np.zeros((1, 1), dtype=np.float32)]

    with pytest.raises(UpstreamError):
        await _reranker(tokenizer_file, Bad()).score("a", ["b", "c"])


def test_invalid_batch_size(tokenizer_file: Path) -> None:
    with pytest.raises(ValidationError):
        _reranker(tokenizer_file, LogitSession(), batch_size=0)


@pytest.mark.parametrize(
    ("name", "variant", "arch", "key", "threshold"),
    [
        ("fp32", "fp32", None, "fp32-sha", 0.1),
        ("int8 arm64", "int8", "arm64", "arm-sha", 0.2),
        ("int8 amd64", "int8", "amd64", "amd-sha", 0.3),
    ],
)
def test_from_manifest_selects_variant(  # noqa: PLR0913, PLR0917
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_file: Path,
    name: str,
    variant: str,
    arch: str | None,
    key: str,
    threshold: float,
) -> None:
    opened: list[Path] = []

    def fake_open(path: Path, threads: int) -> FakeSession:
        opened.append(path)
        return LogitSession()

    monkeypatch.setattr(module, "open_session", fake_open)
    entry = ModelEntry(
        id="m",
        kind="reranker",
        repo="o/m",
        revision="r",
        files={"tokenizer": ModelFile("tokenizer.json", "t", 1)},
        variants={
            "fp32": ModelFile("f.onnx", "fp32-sha", 1, 0.1),
            "int8-arm64": ModelFile("a.onnx", "arm-sha", 1, 0.2),
            "int8-amd64": ModelFile("x.onnx", "amd-sha", 1, 0.3),
        },
    )
    models = tokenizer_file.parent
    (models / "m").mkdir()
    (models / "m" / "tokenizer.json").write_bytes(tokenizer_file.read_bytes())
    rr = OnnxReranker.from_manifest(models, entry, variant=variant, arch=arch)
    assert (rr.model_key, rr.calibrated_threshold) == (key, threshold), name
    assert opened[0].parent == models / "m"


async def test_noop_preserves_order() -> None:
    rr = NoopReranker()
    scores = await rr.score("q", ["x", "y", "z"])
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 3
    assert rr.model_key == "noop"
    assert rr.calibrated_threshold is None
    assert await rr.score("q", []) == []
