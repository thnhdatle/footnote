# SPDX-License-Identifier: Apache-2.0
"""Tenant export."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import orjson
from psycopg.rows import dict_row

from footnote.adapters.postgres.codec import decode_elements, decompress
from footnote.adapters.postgres.session import Conn, Database, translate_errors
from footnote.domain.errors import NotFoundError
from footnote.domain.tenancy import ExportRecord, TenantId

_FETCH = 500

_TENANT = "SELECT id, name, quotas, status, created_at FROM footnote.tenants WHERE id = %(t)s"
_COLLECTIONS = (
    "SELECT c.id, c.name, c.settings, c.language, a.name AS active_space, "
    "t.name AS target_space, c.chunk_count, c.reembed, c.webhook_url, c.status, c.created_at, "
    "c.updated_at FROM footnote.collections c "
    "JOIN footnote.embedding_spaces a ON a.id = c.active_space_id "
    "LEFT JOIN footnote.embedding_spaces t ON t.id = c.target_space_id "
    "WHERE c.tenant_id = %(t)s ORDER BY c.id"
)
_DOCUMENTS = (
    "SELECT id, collection_id, external_id, title, metadata, status, error, "
    "current_version_id, created_at, updated_at FROM footnote.documents "
    "WHERE tenant_id = %(t)s ORDER BY id"
)
_VERSIONS = (
    "SELECT v.id, v.document_id, v.version, encode(v.source_sha256, 'hex') AS source_sha256, "
    "v.media_type, v.bytes, v.pages, v.indexed_title, v.chunking_hash, v.status, v.error, "
    "v.created_at, v.updated_at, {blob} AS original FROM footnote.document_versions v "
    "LEFT JOIN footnote.document_blobs b ON b.tenant_id = v.tenant_id AND b.version_id = v.id "
    "WHERE v.tenant_id = %(t)s ORDER BY v.id"
)
_ELEMENTS = (
    "SELECT id AS version_id, elements FROM footnote.document_versions "
    "WHERE tenant_id = %(t)s AND elements IS NOT NULL ORDER BY id"
)
_CHUNKS = (
    "SELECT id, collection_id, document_id, content_hash, text, heading_path, title, "
    "token_count, fts_config::text AS fts_config FROM footnote.chunks "
    "WHERE tenant_id = %(t)s ORDER BY id"
)
_MEMBERSHIPS = (
    "SELECT version_id, ordinal, chunk_id, page_from, page_to FROM footnote.version_chunks "
    "WHERE tenant_id = %(t)s ORDER BY version_id, ordinal"
)
_VECTORS = (
    "SELECT v.chunk_id, s.name AS space, v.embedding::text AS embedding "
    "FROM footnote.chunk_vectors v JOIN footnote.embedding_spaces s ON s.id = v.space_id "
    "WHERE v.tenant_id = %(t)s ORDER BY v.chunk_id, v.space_id"
)

type Shape = Callable[[dict[str, Any]], dict[str, Any]]


def _same(row: dict[str, Any]) -> dict[str, Any]:
    return row


def _version(row: dict[str, Any]) -> dict[str, Any]:
    original = row.pop("original")
    if original is not None:
        row["original_base64"] = base64.b64encode(decompress(bytes(original))).decode()
    return row


def _elements(row: dict[str, Any]) -> dict[str, Any]:
    elements = decode_elements(bytes(row["elements"]))
    return {"version_id": row["version_id"], "elements": [e.to_dict() for e in elements]}


def _vector(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "embedding": orjson.loads(row["embedding"])}


class PgTenantExport:
    """Streams the rows of a tenant as export records."""

    def __init__(self, admin: Database) -> None:
        self._db = admin

    async def export(
        self, tenant: TenantId, *, include_vectors: bool, include_originals: bool
    ) -> AsyncIterator[ExportRecord]:
        steps: list[tuple[str, str, Shape]] = [
            ("collection", _COLLECTIONS, _same),
            ("document", _DOCUMENTS, _same),
            (
                "version",
                _VERSIONS.format(blob="b.data" if include_originals else "NULL::bytea"),
                _version,
            ),
            ("elements", _ELEMENTS, _elements),
            ("chunk", _CHUNKS, _same),
            ("membership", _MEMBERSHIPS, _same),
        ]
        if include_vectors:
            steps.append(("vector", _VECTORS, _vector))
        async with self._db.pool.connection() as conn, translate_errors(), conn.transaction():
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            await conn.execute(
                "SELECT set_config('statement_timeout', %(t)s, true)",
                {"t": str(self._db.statement_timeout_ms)},
            )
            tenant_row = await _one(conn, tenant)
            yield ExportRecord("tenant", tenant_row)
            for n, (kind, query, shape) in enumerate(steps):
                async for row in _stream(conn, f"footnote_export_{n}", query, tenant):
                    yield ExportRecord(kind, shape(row))


async def _one(conn: Conn, tenant: TenantId) -> dict[str, Any]:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(_TENANT, {"t": tenant})
        row = await cur.fetchone()
    if row is None:
        raise NotFoundError("tenant not found")
    return row


async def _stream(
    conn: Conn, name: str, query: str, tenant: TenantId
) -> AsyncIterator[dict[str, Any]]:
    async with conn.cursor(name=name, row_factory=dict_row) as cur:
        await cur.execute(query, {"t": tenant})
        while rows := await cur.fetchmany(_FETCH):
            for row in rows:
                yield row


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_tenantexporter(db: Database) -> ports.TenantExporter:
        return PgTenantExport(db)
