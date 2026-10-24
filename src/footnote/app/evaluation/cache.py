# SPDX-License-Identifier: Apache-2.0
"""SQLite embedding cache for evaluation runs."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson

from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.ports import Embedder

DOCUMENT = "document"
QUERY = "query"


def cache_key_for_space(space: EmbeddingSpace, model_sha256: str | None = None) -> str:
    if space.provider == "onnx_local":
        if not model_sha256:
            msg = "onnx_local cache key needs the model file sha256"
            raise ValueError(msg)
        return f"onnx_local/{model_sha256}"
    return f"{space.provider}/{space.model}/{space.dimensions}"


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".sqlite")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _round(vector: Sequence[float]) -> list[float]:
    return [float(x) for x in np.asarray(vector, dtype="<f4")]


class CachingEmbedder:
    """Embedder that serves repeated texts from a SQLite cache."""

    def __init__(
        self, inner: Embedder, cache_dir: Path, key: str, *, cache_queries: bool = True
    ) -> None:
        self._inner = inner
        self.key = key
        self._cache_queries = cache_queries
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_path(cache_dir, key)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (kind TEXT NOT NULL, text_sha256 TEXT NOT NULL, "
            "vector BLOB NOT NULL, PRIMARY KEY (kind, text_sha256))"
        )
        self._db.execute("INSERT OR IGNORE INTO meta VALUES ('key', ?)", (key,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _get(self, kind: str, digests: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(digests), 500):
            chunk = digests[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT text_sha256, vector FROM vectors WHERE kind = ? "  # noqa: S608 - marks only
                f"AND text_sha256 IN ({marks})",
                (kind, *chunk),
            )
            for digest, blob in rows:
                found[digest] = [float(x) for x in np.frombuffer(blob, dtype="<f4")]
        return found

    def _put(self, kind: str, items: list[tuple[str, list[float]]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
            [(kind, d, np.asarray(v, dtype="<f4").tobytes()) for d, v in items],
        )
        self._db.commit()

    async def _embed(self, kind: str, texts: Sequence[str]) -> list[list[float]]:
        digests = [_digest(t) for t in texts]
        found = self._get(kind, list(dict.fromkeys(digests)))
        missing: dict[str, str] = {}
        for text, digest in zip(texts, digests, strict=True):
            if digest not in found:
                missing.setdefault(digest, text)
        if missing:
            order = list(missing)
            if kind == QUERY:
                raw = [await self._inner.embed_query(missing[d]) for d in order]
            else:
                raw = await self._inner.embed_documents([missing[d] for d in order])
            fresh = [(d, _round(v)) for d, v in zip(order, raw, strict=True)]
            self._put(kind, fresh)
            found.update(fresh)
        return [list(found[d]) for d in digests]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(DOCUMENT, texts)

    async def embed_query(self, text: str) -> list[float]:
        if not self._cache_queries:
            return _round(await self._inner.embed_query(text))
        return (await self._embed(QUERY, [text]))[0]


def _marker(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / "warm" / f"{dataset}.json"


def mark_warm(cache_dir: Path, dataset: str, key: str, texts: int) -> None:
    path = _marker(cache_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": dataset,
        "key": key,
        "texts": texts,
        "written_at": datetime.now(UTC).isoformat(),
    }
    path.write_bytes(orjson.dumps(data))


def is_warm(cache_dir: Path, dataset: str) -> bool:
    path = _marker(cache_dir, dataset)
    if not path.is_file():
        return False
    try:
        key = str(orjson.loads(path.read_bytes())["key"])
    except (orjson.JSONDecodeError, KeyError):
        return False
    return cache_path(cache_dir, key).is_file()


__all__ = [
    "CachingEmbedder",
    "cache_key_for_space",
    "cache_path",
    "is_warm",
    "mark_warm",
]
