# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from footnote.adapters.embedders import OnnxEmbedder
from footnote.adapters.models import FileStatus, ModelEntry, load_manifest, pull_model, verify_model
from footnote.adapters.rerankers import OnnxReranker

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
EMBEDDER_TOLERANCE = 1e-4
RERANKER_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def models_dir() -> Path:
    default = Path.home() / ".cache" / "rayld" / "footnote" / "models"
    return Path(os.environ.get("FOOTNOTE_MODELS_DIR", default))


def _ready(models_dir: Path, model_id: str, variants: list[str]) -> ModelEntry:
    entry = load_manifest(ROOT / "models" / "manifest.yaml").get(model_id)
    pull_model(models_dir, entry, variants)
    checks = verify_model(models_dir, entry, variants)
    assert all(c.status is FileStatus.OK for c in checks), checks
    return entry


def _reference(model_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (ROOT / "tests" / "reference" / f"{model_id}.json").read_text()
    )
    return data


async def test_embedder_matches_reference(models_dir: Path) -> None:
    ref = _reference("bge-small-en-v1.5")
    entry = _ready(models_dir, ref["model_id"], [])
    assert entry.revision == ref["revision"]
    embedder = OnnxEmbedder.from_manifest(models_dir, entry)
    got = np.array(await embedder.embed_documents(ref["texts"]))
    want = np.array(ref["embeddings"])
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= EMBEDDER_TOLERANCE


async def test_reranker_fp32_matches_reference(models_dir: Path) -> None:
    ref = _reference("ms-marco-MiniLM-L6-v2")
    entry = _ready(models_dir, ref["model_id"], ["fp32"])
    assert entry.revision == ref["revision"]
    reranker = OnnxReranker.from_manifest(models_dir, entry, variant="fp32")
    got = [await reranker.score(q, [p]) for q, p in ref["pairs"]]
    flat = np.array([g[0] for g in got])
    assert np.abs(flat - np.array(ref["logits"])).max() <= RERANKER_TOLERANCE
