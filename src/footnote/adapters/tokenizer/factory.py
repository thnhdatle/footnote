# SPDX-License-Identifier: Apache-2.0
"""Token counter factory over the model manifest."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from footnote.adapters.models.manifest import Manifest, local_path
from footnote.adapters.tokenizer.counter import HfTokenCounter
from footnote.domain.errors import UnavailableError

TOKENIZER_FILE = "tokenizer"


def tokenizer_ids(manifest: Manifest) -> frozenset[str]:
    return frozenset(m.id for m in manifest.models.values() if TOKENIZER_FILE in m.files)


class ManifestTokenizers:
    """Loads and caches one token counter per tokenizer id."""

    def __init__(self, manifest: Manifest, models_dir: Path) -> None:
        self._manifest = manifest
        self._models_dir = models_dir
        self._cache: dict[str, HfTokenCounter] = {}
        self._lock = threading.Lock()

    def for_tokenizer(self, tokenizer_id: str) -> HfTokenCounter:
        """Returns the counter for a tokenizer id."""
        with self._lock:
            cached = self._cache.get(tokenizer_id)
            if cached is None:
                cached = self._load(tokenizer_id)
                self._cache[tokenizer_id] = cached
            return cached

    def _load(self, tokenizer_id: str) -> HfTokenCounter:
        entry = self._manifest.models.get(tokenizer_id)
        if entry is None or TOKENIZER_FILE not in entry.files:
            raise UnavailableError(f"tokenizer {tokenizer_id!r} is not in the model manifest")
        path = local_path(self._models_dir, entry, entry.files[TOKENIZER_FILE])
        if not path.is_file():
            raise UnavailableError(
                f"tokenizer {tokenizer_id!r} is missing at {path}; run `footnote models pull`"
            )
        return HfTokenCounter.from_file(path)


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms(manifest: Manifest) -> ports.TokenCounterFactory:
        return ManifestTokenizers(manifest, Path())
