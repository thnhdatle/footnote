# SPDX-License-Identifier: Apache-2.0
"""SHA-256 verification of local model files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from footnote.adapters.models.manifest import ModelEntry, ModelFile, local_path

_CHUNK = 1 << 20


class FileStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class FileCheck:
    model_id: str
    name: str
    path: Path
    status: FileStatus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, file: ModelFile) -> FileStatus:
    if not path.is_file():
        return FileStatus.MISSING
    if path.stat().st_size != file.size or sha256_file(path) != file.sha256:
        return FileStatus.MISMATCH
    return FileStatus.OK


def selected_files(entry: ModelEntry, variants: list[str] | None = None) -> dict[str, ModelFile]:
    files = dict(entry.files)
    for key in variants or []:
        files[key] = entry.variants[key]
    return files


def verify_model(
    models_dir: Path, entry: ModelEntry, variants: list[str] | None = None
) -> list[FileCheck]:
    """Verifies the files of one model against the manifest."""
    return [
        FileCheck(entry.id, name, path, check_file(path, file))
        for name, file in selected_files(entry, variants).items()
        for path in [local_path(models_dir, entry, file)]
    ]
