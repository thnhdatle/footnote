# SPDX-License-Identifier: Apache-2.0
"""Readiness of the local model files."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from footnote.adapters.models.manifest import ModelEntry, ModelFile, local_path
from footnote.adapters.models.verify import FileStatus, check_file, selected_files


@dataclass(frozen=True, slots=True)
class ModelNeed:
    entry: ModelEntry
    keys: tuple[str, ...] = field(default=())

    def files(self) -> dict[str, ModelFile]:
        if not self.keys:
            return selected_files(self.entry)
        return {k: self.entry.files.get(k) or self.entry.variants[k] for k in self.keys}


@dataclass(frozen=True, slots=True)
class _Pending:
    model_id: str
    name: str
    path: Path
    file: ModelFile


class ModelReadiness:
    """Checks that the local model files are present and verified."""

    def __init__(self, models_dir: Path, needs: Sequence[ModelNeed]) -> None:
        self._pending = [
            _Pending(need.entry.id, name, local_path(models_dir, need.entry, file), file)
            for need in needs
            for name, file in need.files().items()
        ]
        self._lock = threading.Lock()

    def check(self) -> list[str]:
        with self._lock:
            problems: list[str] = []
            still: list[_Pending] = []
            for item in self._pending:
                status = check_file(item.path, item.file)
                if status is FileStatus.OK:
                    continue
                still.append(item)
                problems.append(f"{item.model_id} {item.name}: {status.value} ({item.path})")
            self._pending = still
            return problems

    async def problems(self) -> list[str]:
        return await asyncio.to_thread(self.check)
