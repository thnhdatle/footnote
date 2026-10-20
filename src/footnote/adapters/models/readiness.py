# SPDX-License-Identifier: Apache-2.0
"""Readiness of the local model files: present and SHA-256-verified.

A file that verified once is not hashed again; files still missing or mismatched are
re-checked on every probe, so `/readyz` turns ready after `footnote models pull` without a
restart. Hashing runs in a worker thread under a lock (one verification at a time).
"""

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
    """A manifest model the configuration uses and the file keys it needs.

    Keys name plain files (`model`, `tokenizer`) or flattened variants (`int8-arm64`);
    an empty `keys` means every plain file of the entry.
    """

    entry: ModelEntry
    keys: tuple[str, ...] = field(default=())

    def files(self) -> dict[str, ModelFile]:
        """The selected files by key (KeyError for a key the entry does not have)."""
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
    """Readiness probe over the files of `needs` below `models_dir`."""

    def __init__(self, models_dir: Path, needs: Sequence[ModelNeed]) -> None:
        self._pending = [
            _Pending(need.entry.id, name, local_path(models_dir, need.entry, file), file)
            for need in needs
            for name, file in need.files().items()
        ]
        self._lock = threading.Lock()

    def check(self) -> list[str]:
        """Problems of the files not yet verified (synchronous; hashes files)."""
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
        """`check` in a worker thread (for `/readyz`)."""
        return await asyncio.to_thread(self.check)
