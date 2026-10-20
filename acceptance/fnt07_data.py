# SPDX-License-Identifier: Apache-2.0
"""FNT-07 data preparation: 100k TREC-COVID chunks bulk-loaded with binary COPY.

The rows are what ingest writes for one plain-text upload per corpus document
(``upload_request`` of the eval harness, the shipped chunker, bge-small vectors from the
evaluation embedding cache): ``documents``, ``document_versions``, ``document_blobs``,
``chunks``, ``version_chunks``, ``chunk_vectors``, plus ``collections.chunk_count``. The
HNSW index of the space is dropped for the load and recreated from its own definition
afterwards, so the end state equals the one registration plus ingest would leave.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import numpy as np
import orjson
import psycopg

from acceptance.harness import (
    ADMIN_DSN,
    ADMIN_KEY,
    APP_DSN,
    LOG_DIR,
    SUPERUSER_DSN,
    Instance,
    connect,
    wait_ready,
)
from footnote.adapters.embedders.onnx_local import OnnxEmbedder
from footnote.adapters.models.manifest import load_manifest
from footnote.adapters.parsers import media
from footnote.adapters.parsers.parse import parse_bytes
from footnote.adapters.postgres.codec import compress
from footnote.adapters.postgres.session import new_id
from footnote.adapters.tokenizer.counter import HfTokenCounter
from footnote.app.evaluation.cache import (
    CachingEmbedder,
    _digest,
    cache_key_for_space,
    cache_path,
    mark_warm,
)
from footnote.app.evaluation.corpus import upload_request
from footnote.app.evaluation.datasets import CorpusDoc, dataset_dir
from footnote.domain.chunking import chunk_elements
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.settings import ChunkingSettings

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path.home() / ".cache/rayld/footnote/models"
EVAL_CACHE_DIR = Path.home() / ".cache/rayld/footnote/eval"
EMBED_MODEL = "bge-small-en-v1.5"
DIMENSIONS = 384
EMBED_BATCH = 256
LANGUAGE = "english"
SERVER_START_TIMEOUT = 240.0
STOP_TIMEOUT = 60.0

DOC_COLUMNS = (
    "tenant_id, id, collection_id, external_id, title, status, current_version_id",
    ["uuid", "uuid", "uuid", "text", "text", "text", "uuid"],
)
VERSION_COLUMNS = (
    "tenant_id, id, document_id, version, source_sha256, media_type, bytes, indexed_title, "
    "chunking_hash, status",
    ["uuid", "uuid", "uuid", "int4", "bytea", "text", "int8", "text", "text", "text"],
)
BLOB_COLUMNS = ("tenant_id, version_id, data", ["uuid", "uuid", "bytea"])
CHUNK_COLUMNS = (
    "tenant_id, id, collection_id, document_id, content_hash, text, heading_path, title, "
    "embedded_text, token_count, fts_config",
    ["uuid", "uuid", "uuid", "uuid", "text", "text", "text[]", "text", "text", "int4", "oid"],
)
MEMBER_COLUMNS = (
    "tenant_id, version_id, ordinal, chunk_id",
    ["uuid", "uuid", "int4", "uuid"],
)
VECTOR_COLUMNS = (
    "tenant_id, chunk_id, space_id, collection_id, embedding",
    ["uuid", "uuid", "int4", "uuid", "bytea"],
)


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """One distinct chunk of a document."""

    id: UUID
    content_hash: str
    text: str
    heading_path: list[str]
    embedded_text: str
    token_count: int


@dataclass(slots=True)
class DocRows:
    """A document with its version and chunks (memberships index into ``chunks``)."""

    doc_id: UUID
    version_id: UUID
    external_id: str
    title: str
    sha256: bytes
    data: bytes
    chunks: list[ChunkRow]
    members: list[int]


def tokenizer_counter() -> HfTokenCounter:
    """The bge-small tokenizer the shipped chunker counts with."""
    return HfTokenCounter.from_file(MODELS_DIR / EMBED_MODEL / "tokenizer.json")


def prepare_documents(limit: int, counter: HfTokenCounter | None = None) -> list[DocRows]:
    """Chunk TREC-COVID documents in corpus file order until ``limit`` chunks exist.

    The last document is cut so the total is exactly ``limit`` distinct chunks.
    """
    counter = counter or tokenizer_counter()
    settings = ChunkingSettings()
    out: list[DocRows] = []
    total = 0
    with (dataset_dir("trec-covid") / "corpus.jsonl").open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = orjson.loads(line)
            doc = CorpusDoc(
                str(row["_id"]), str(row.get("title") or ""), str(row.get("text") or "")
            )
            upload = upload_request(doc)
            title = upload.title or ""
            elements = parse_bytes(media.sniff(upload.data, upload.filename), upload.data)
            drafts = chunk_elements(elements, title, settings, counter)
            chunks: list[ChunkRow] = []
            index: dict[str, int] = {}
            members: list[int] = []
            for draft in drafts:
                if draft.content_hash not in index:
                    if total + len(chunks) >= limit:
                        break
                    index[draft.content_hash] = len(chunks)
                    chunks.append(
                        ChunkRow(
                            new_id(),
                            draft.content_hash,
                            draft.text,
                            list(draft.heading_path),
                            draft.embedded_text,
                            draft.token_count,
                        )
                    )
                members.append(index[draft.content_hash])
            if chunks:
                total += len(chunks)
                out.append(
                    DocRows(
                        new_id(),
                        new_id(),
                        doc.doc_id,
                        title,
                        hashlib.sha256(upload.data).digest(),
                        upload.data,
                        chunks,
                        members,
                    )
                )
            if total >= limit:
                break
    if total < limit:
        msg = f"TREC-COVID yields only {total} chunks, need {limit}"
        raise RuntimeError(msg)
    return out


def embed_chunks(docs: Sequence[DocRows], models_dir: Path = MODELS_DIR) -> np.ndarray:
    """Vectors of every chunk in document order, from the eval cache (misses computed)."""
    manifest = load_manifest(ROOT / "models" / "manifest.yaml")
    entry = manifest.get(EMBED_MODEL)
    space = EmbeddingSpace(0, "fnt07", "onnx_local", EMBED_MODEL, DIMENSIONS)
    key = cache_key_for_space(space, entry.files["model"].sha256)
    inner = OnnxEmbedder.from_manifest(models_dir, entry)
    cache = CachingEmbedder(inner, EVAL_CACHE_DIR, key)
    texts = [c.embedded_text for d in docs for c in d.chunks]

    async def run() -> np.ndarray:
        vectors = np.empty((len(texts), DIMENSIONS), dtype=np.float32)
        for start in range(0, len(texts), EMBED_BATCH):
            batch = await cache.embed_documents(texts[start : start + EMBED_BATCH])
            vectors[start : start + len(batch)] = np.asarray(batch, dtype=np.float32)
        return vectors

    try:
        vectors = asyncio.run(run())
        mark_warm(EVAL_CACHE_DIR, "trec-covid", key, len(texts))
    finally:
        cache.close()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


def _vector_bytes(vector: np.ndarray) -> bytes:
    """pgvector binary wire format: int16 dim, int16 unused, big-endian float4 values."""
    return struct.pack(">HH", len(vector), 0) + vector.astype(">f4").tobytes()


def _copy(
    cur: psycopg.Cursor[Any],
    table: str,
    spec: tuple[str, list[str]],
    rows: Iterable[Sequence[Any]],
) -> int:
    columns, types = spec
    count = 0
    with cur.copy(f"COPY footnote.{table} ({columns}) FROM STDIN (FORMAT BINARY)") as copy:
        copy.set_types(types)
        for row in rows:
            copy.write_row(row)
            count += 1
    return count


@dataclass(frozen=True, slots=True)
class LoadIds:
    """Identifiers the load needs from the API-created tenant and collection."""

    tenant: UUID
    collection: UUID
    space: int
    chunking_hash: str
    fts_oid: int


def _rows(docs: Sequence[DocRows], vectors: np.ndarray, ids: LoadIds) -> Iterator[tuple[str, Any]]:
    yield (
        "documents",
        (
            (ids.tenant, d.doc_id, ids.collection, d.external_id, d.title, "ready", d.version_id)
            for d in docs
        ),
    )
    yield (
        "document_versions",
        (
            (
                ids.tenant,
                d.version_id,
                d.doc_id,
                1,
                d.sha256,
                media.TEXT,
                len(d.data),
                d.title,
                ids.chunking_hash,
                "ready",
            )
            for d in docs
        ),
    )
    yield "document_blobs", ((ids.tenant, d.version_id, compress(d.data)) for d in docs)
    yield (
        "chunks",
        (
            (
                ids.tenant,
                c.id,
                ids.collection,
                d.doc_id,
                c.content_hash,
                c.text,
                c.heading_path,
                d.title,
                c.embedded_text,
                c.token_count,
                ids.fts_oid,
            )
            for d in docs
            for c in d.chunks
        ),
    )
    yield (
        "version_chunks",
        (
            (ids.tenant, d.version_id, ordinal, d.chunks[member].id)
            for d in docs
            for ordinal, member in enumerate(d.members)
        ),
    )
    yield (
        "chunk_vectors",
        (
            (ids.tenant, c.id, ids.space, ids.collection, _vector_bytes(vectors[position]))
            for position, c in enumerate(c for d in docs for c in d.chunks)
        ),
    )


def bulk_load(
    dsn: str, tenant: str, collection: str, docs: Sequence[DocRows], vectors: np.ndarray
) -> dict[str, Any]:
    """COPY everything over the superuser DSN in one transaction; returns counts and timings."""
    started = time.monotonic()
    with psycopg.connect(dsn) as conn:
        space, fts_oid = conn.execute(
            "SELECT c.active_space_id, ('english'::regconfig)::oid::int "
            "FROM footnote.collections c "
            "WHERE c.tenant_id = %s AND c.id = %s AND c.language = 'english'",
            (tenant, collection),
        ).fetchone() or (None, None)
        if space is None:
            msg = "collection not found or not an english collection"
            raise RuntimeError(msg)
        index_name, index_def = conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'footnote' "
            "AND tablename = 'chunk_vectors' AND indexname = %s",
            (f"chunk_vectors_hnsw_s{space}",),
        ).fetchone() or (None, None)
        if index_def is None:
            msg = f"HNSW index chunk_vectors_hnsw_s{space} missing; register the space first"
            raise RuntimeError(msg)
        ids = LoadIds(
            UUID(str(tenant)),
            UUID(str(collection)),
            int(space),
            ChunkingSettings().settings_hash(),
            int(fts_oid),
        )
        counts: dict[str, int] = {}
        conn.execute(f'DROP INDEX footnote."{index_name}"')
        with conn.cursor() as cur:
            for table, rows in _rows(docs, vectors, ids):
                spec = {
                    "documents": DOC_COLUMNS,
                    "document_versions": VERSION_COLUMNS,
                    "document_blobs": BLOB_COLUMNS,
                    "chunks": CHUNK_COLUMNS,
                    "version_chunks": MEMBER_COLUMNS,
                    "chunk_vectors": VECTOR_COLUMNS,
                }[table]
                counts[table] = _copy(cur, table, spec, rows)
            cur.execute(
                "UPDATE footnote.collections SET chunk_count = %s, updated_at = now() "
                "WHERE tenant_id = %s AND id = %s",
                (counts["chunks"], tenant, collection),
            )
        copied = time.monotonic()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(index_def)  # same definition registration used
        for table in ("chunks", "version_chunks", "chunk_vectors", "documents"):
            conn.execute(f"VACUUM (ANALYZE) footnote.{table}")
    done = time.monotonic()
    return {
        "rows": counts,
        "copy_seconds": round(copied - started, 1),
        "index_and_analyze_seconds": round(done - copied, 1),
    }


def free_port() -> int:
    """An unused loopback TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def queries() -> list[str]:
    """The `text` field of the 50 TREC-COVID queries, in file order."""
    path = dataset_dir("trec-covid") / "queries.jsonl"
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    return [str(row["text"]) for row in rows]


@contextlib.contextmanager
def native_server(
    extra_env: Mapping[str, str] | None = None, log_name: str = "fnt07-server.log"
) -> Iterator[Instance]:
    """`footnote serve` as a subprocess on loopback ports against the compose Postgres."""
    data_port, admin_port = free_port(), free_port()
    env = {
        **os.environ,
        "FOOTNOTE_SERVER_LISTEN": f"127.0.0.1:{data_port}",
        "FOOTNOTE_ADMIN_LISTEN": f"127.0.0.1:{admin_port}",
        "FOOTNOTE_DATABASE_URL": APP_DSN,
        "FOOTNOTE_ADMIN_DATABASE_URL": ADMIN_DSN,
        "FOOTNOTE_AUTH_ADMIN_KEY": ADMIN_KEY,
        "FOOTNOTE_LOG_LEVEL": "warn",
        "FOOTNOTE_MODELS_MANIFEST": str(ROOT / "models" / "manifest.yaml"),
        **(extra_env or {}),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / log_name).open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "footnote", "serve"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            inst = Instance(f"http://127.0.0.1:{data_port}", f"http://127.0.0.1:{admin_port}", "")
            wait_ready(inst.admin_url, timeout=SERVER_START_TIMEOUT)
            yield inst
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def drop_stale_tenants(prefix: str) -> None:
    """Remove tenants a crashed earlier run left behind (cascade deletes their rows)."""
    with connect(SUPERUSER_DSN) as conn:
        conn.execute("DELETE FROM footnote.tenants WHERE name LIKE %s", (f"{prefix}%",))


def timed_run(
    client: httpx.Client, path: str, texts: Sequence[str], count: int, *, offset: int = 0
) -> list[float]:
    """`count` sequential hybrid+rerank searches (k 8); client wall time in ms per request."""
    body: dict[str, Any] = {"mode": "hybrid", "rerank": True, "k": 8}
    out: list[float] = []
    for i in range(count):
        start = time.perf_counter()
        resp = client.post(path, json={**body, "query": texts[(offset + i) % len(texts)]})
        elapsed = (time.perf_counter() - start) * 1000.0
        assert resp.status_code == 200, f"search failed: {resp.status_code} {resp.text[:300]}"
        out.append(elapsed)
    return out


def warm_marker(limit: int = 100_000) -> int:
    """Check every FNT-07 chunk vector is cached (nothing is embedded), then mark warm.

    Returns the number of texts checked; raises when any vector is missing.
    """
    docs = prepare_documents(limit)
    manifest = load_manifest(ROOT / "models" / "manifest.yaml")
    entry = manifest.get(EMBED_MODEL)
    space = EmbeddingSpace(0, "fnt07", "onnx_local", EMBED_MODEL, DIMENSIONS)
    key = cache_key_for_space(space, entry.files["model"].sha256)
    path = cache_path(EVAL_CACHE_DIR, key)
    if not path.is_file():
        msg = f"embedding cache {path} does not exist"
        raise RuntimeError(msg)
    digests = {_digest(c.embedded_text) for d in docs for c in d.chunks}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        cached = {
            row[0] for row in db.execute("SELECT text_sha256 FROM vectors WHERE kind = 'document'")
        }
    missing = len(digests - cached)
    if missing:
        msg = f"{missing} of {len(digests)} FNT-07 chunk vectors are missing from {path}"
        raise RuntimeError(msg)
    mark_warm(EVAL_CACHE_DIR, "trec-covid", key, limit)
    return len(digests)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", action="store_true", help="mark the trec-covid cache warm")
    args = parser.parse_args()
    if not args.warm:
        parser.error("nothing to do; pass --warm")
    print(f"checked {warm_marker()} cached vectors; trec-covid marked warm")
