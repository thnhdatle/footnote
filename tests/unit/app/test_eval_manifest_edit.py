# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import yaml

from footnote.app.evaluation.manifest_edit import (
    ManifestEditError,
    set_file_threshold,
    set_model_threshold,
    stored_threshold,
)

MANIFEST = """\
# header comment
models:
  - id: emb
    kind: embedder
    files:
      model: {path: m.onnx, sha256: aaaa, bytes: 1}
    dimensions: 384
    calibrated_threshold: 0.35   # vector gate
  - id: rr
    kind: reranker
    files:
      tokenizer: {path: t.json, sha256: tttt, bytes: 1}
    variants:
      fp32: {path: f.onnx, sha256: ffff, bytes: 2, calibrated_threshold: 0.35}
      int8:
        arm64: {path: a.onnx, sha256: bbbb, bytes: 3}
    calibrated_threshold: 0.99
"""


def _models(text: str) -> dict[str, dict[str, object]]:
    return {m["id"]: m for m in yaml.safe_load(text)["models"]}


def test_set_file_threshold_replaces_or_adds_the_flow_key() -> None:
    text = set_file_threshold(MANIFEST, "ffff", 0.4123)
    text = set_file_threshold(text, "bbbb", 0.05)
    rr = _models(text)["rr"]
    variants = rr["variants"]
    assert isinstance(variants, dict)
    assert variants["fp32"]["calibrated_threshold"] == 0.4123
    assert variants["int8"]["arm64"]["calibrated_threshold"] == 0.05
    assert "fp32: {path: f.onnx, sha256: ffff, bytes: 2, calibrated_threshold: 0.4123}" in text
    assert text.startswith("# header comment\n")
    assert _models(text)["emb"]["calibrated_threshold"] == 0.35


def test_set_file_threshold_needs_exactly_one_match() -> None:
    with pytest.raises(ManifestEditError, match="found 0"):
        set_file_threshold(MANIFEST, "nope", 0.1)
    doubled = MANIFEST + "      x: {path: y, sha256: ffff}\n"
    with pytest.raises(ManifestEditError, match="found 2"):
        set_file_threshold(doubled, "ffff", 0.1)


def test_set_model_threshold_edits_only_that_entry_and_keeps_comments() -> None:
    text = set_model_threshold(MANIFEST, "emb", 0.7071)
    assert "    calibrated_threshold: 0.7071   # vector gate\n" in text
    models = _models(text)
    assert models["emb"]["calibrated_threshold"] == 0.7071
    assert models["rr"]["calibrated_threshold"] == 0.99
    assert models["rr"]["variants"]["fp32"]["calibrated_threshold"] == 0.35  # type: ignore[index]
    with pytest.raises(ManifestEditError, match="no model"):
        set_model_threshold(MANIFEST, "missing", 0.1)
    bare = MANIFEST.replace("    calibrated_threshold: 0.35   # vector gate\n", "")
    with pytest.raises(ManifestEditError, match="no calibrated_threshold"):
        set_model_threshold(bare, "emb", 0.1)


@pytest.mark.parametrize(
    ("value", "stored"), [(0.35, 0.35), (0.123456, 0.1234), (0.99999, 0.9999), (0.0, 0.0)]
)
def test_stored_threshold_rounds_down_to_four_decimals(value: float, stored: float) -> None:
    assert stored_threshold(value) == stored
    assert stored_threshold(value) <= value
