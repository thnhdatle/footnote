# SPDX-License-Identifier: Apache-2.0
"""Parsing of the pinned model manifest."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from footnote.domain.errors import NotFoundError, ValidationError

ARM64 = "arm64"
AMD64 = "amd64"


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    sha256: str
    size: int
    calibrated_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class ModelEntry:
    id: str
    kind: str
    repo: str
    revision: str
    files: dict[str, ModelFile]
    variants: dict[str, ModelFile] = field(default_factory=dict)
    dimensions: int | None = None
    pooling: str = "cls"
    normalise: bool = True
    query_prefix: str = ""
    max_tokens: int = 512
    calibrated_threshold: float | None = None


def host_arch() -> str:
    return ARM64 if platform.machine().lower() in {"arm64", "aarch64"} else AMD64


def _file(raw: dict[str, Any]) -> ModelFile:
    threshold = raw.get("calibrated_threshold")
    return ModelFile(
        path=str(raw["path"]),
        sha256=str(raw["sha256"]).lower(),
        size=int(raw["bytes"]),
        calibrated_threshold=None if threshold is None else float(threshold),
    )


def _variants(raw: dict[str, Any]) -> dict[str, ModelFile]:
    out: dict[str, ModelFile] = {}
    for name, value in raw.items():
        if "path" in value:
            out[name] = _file(value)
        else:
            out.update({f"{name}-{arch}": _file(v) for arch, v in value.items()})
    return out


def _entry(raw: dict[str, Any]) -> ModelEntry:
    threshold = raw.get("calibrated_threshold")
    return ModelEntry(
        id=str(raw["id"]),
        kind=str(raw["kind"]),
        repo=str(raw["repo"]),
        revision=str(raw["revision"]),
        files={k: _file(v) for k, v in raw.get("files", {}).items()},
        variants=_variants(raw.get("variants", {})),
        dimensions=raw.get("dimensions"),
        pooling=str(raw.get("pooling", "cls")),
        normalise=bool(raw.get("normalise", True)),
        query_prefix=str(raw.get("query_prefix", "")),
        max_tokens=int(raw.get("max_tokens", 512)),
        calibrated_threshold=None if threshold is None else float(threshold),
    )


@dataclass(frozen=True, slots=True)
class Manifest:
    """The pinned models by id."""

    models: dict[str, ModelEntry]

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self.models[model_id]
        except KeyError:
            raise NotFoundError(f"model {model_id!r} is not in the manifest") from None


def load_manifest(path: Path) -> Manifest:
    """Reads and validates a manifest file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = [_entry(e) for e in raw["models"]]
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidationError("models.manifest", f"cannot load {path}: {exc}") from exc
    return Manifest({e.id: e for e in entries})


def reranker_variant_key(variant: str, arch: str | None = None) -> str:
    if variant == "fp32":
        return "fp32"
    if variant == "int8":
        return f"int8-{arch or host_arch()}"
    raise ValidationError("rerank.variant", f"{variant!r} must be int8 or fp32")


def local_path(models_dir: Path, entry: ModelEntry, file: ModelFile) -> Path:
    return models_dir / entry.id / file.path
