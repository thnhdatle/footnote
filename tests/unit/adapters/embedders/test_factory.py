# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import footnote.adapters.embedders.factory as module
from footnote.adapters.embedders import (
    DefaultEmbedderFactory,
    OnnxEmbedder,
    OpenAICompatDefaults,
    OpenAICompatEmbedder,
)
from footnote.adapters.models.manifest import Manifest, ModelEntry, ModelFile
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import UnavailableError

FILE = ModelFile("x", "s", 1)


def _manifest() -> Manifest:
    files = {"model": FILE, "tokenizer": FILE}
    return Manifest(
        {
            "emb": ModelEntry("emb", "embedder", "o/e", "r", files, dimensions=4),
            "rr": ModelEntry("rr", "reranker", "o/r", "r", {}),
        }
    )


def _space(provider: str, model: str, space_id: int = 1) -> EmbeddingSpace:
    return EmbeddingSpace(space_id, f"s{space_id}", provider, model, 4)


def _factory(tmp_path: Path) -> DefaultEmbedderFactory:
    return DefaultEmbedderFactory(
        _manifest(), tmp_path, OpenAICompatDefaults(base_url="http://p/v1")
    )


def test_onnx_space_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    built: list[str] = []

    def fake(models_dir: Path, entry: ModelEntry, **kw: Any) -> OnnxEmbedder:
        built.append(entry.id)
        return object()  # type: ignore[return-value]

    monkeypatch.setattr(OnnxEmbedder, "from_manifest", staticmethod(fake))
    factory = _factory(tmp_path)
    first = factory.for_space(_space("onnx_local", "emb"))
    assert factory.for_space(_space("onnx_local", "emb")) is first
    assert built == ["emb"]


def test_openai_space(tmp_path: Path) -> None:
    emb = _factory(tmp_path).for_space(_space("openai_compat", "remote", 2))
    assert isinstance(emb, OpenAICompatEmbedder)


@pytest.mark.parametrize(
    ("name", "provider", "model", "defaults_url"),
    [
        ("unknown provider", "mystery", "emb", "http://p/v1"),
        ("model missing", module.PROVIDER_ONNX, "nope", "http://p/v1"),
        ("model is not an embedder", module.PROVIDER_ONNX, "rr", "http://p/v1"),
        ("no base url", module.PROVIDER_OPENAI, "remote", ""),
    ],
)
def test_unavailable(
    tmp_path: Path, name: str, provider: str, model: str, defaults_url: str
) -> None:
    factory = DefaultEmbedderFactory(
        _manifest(), tmp_path, OpenAICompatDefaults(base_url=defaults_url)
    )
    with pytest.raises(UnavailableError):
        factory.for_space(_space(provider, model))
