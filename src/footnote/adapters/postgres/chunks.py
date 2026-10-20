# SPDX-License-Identifier: Apache-2.0
"""Read one chunk of a current version (`GET /v1/collections/{id}/chunks/{chunk_id}`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql

from footnote.adapters.postgres.session import Database, fetch_all
from footnote.domain.documents import (
    ChunkDetail,
    ChunkId,
    ChunkPlacement,
    CollectionId,
    DocumentId,
)
from footnote.domain.errors import NotFoundError
from footnote.domain.tenancy import TenantId

CHUNK_NOT_FOUND = "chunk_not_found"


class PgChunks:
    """Chunk reads under the tenant's RLS setting; other tenants' chunks look absent."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def chunk(
        self, tenant: TenantId, collection_id: CollectionId, chunk_id: ChunkId
    ) -> ChunkDetail:
        """The chunk with its current-version memberships; `chunk_not_found` otherwise."""
        query = sql.SQL(
            "SELECT c.id, c.collection_id, c.document_id, d.title AS document_title, c.text, "
            "c.heading_path, c.token_count, v.version, vc.ordinal, vc.page_from, vc.page_to "
            "FROM footnote.chunks c "
            "JOIN footnote.collections col ON col.tenant_id = c.tenant_id "
            "AND col.id = c.collection_id AND col.status = 'active' "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
            "AND vc.version_id = d.current_version_id AND vc.chunk_id = c.id "
            "JOIN footnote.document_versions v ON v.tenant_id = vc.tenant_id "
            "AND v.id = vc.version_id "
            "WHERE {} AND c.collection_id = %(cid)s AND c.id = %(id)s ORDER BY vc.ordinal"
        ).format(self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(
                conn, query, {"tenant": tenant, "cid": collection_id, "id": chunk_id}
            )
        if not rows:
            raise NotFoundError("chunk not found in a current version", code=CHUNK_NOT_FOUND)
        first = rows[0]
        return ChunkDetail(
            id=ChunkId(first["id"]),
            collection_id=CollectionId(first["collection_id"]),
            document_id=DocumentId(first["document_id"]),
            document_title=first["document_title"],
            text=first["text"],
            heading_path=tuple(first["heading_path"]),
            token_count=int(first["token_count"]),
            memberships=tuple(
                ChunkPlacement(int(r["version"]), int(r["ordinal"]), r["page_from"], r["page_to"])
                for r in rows
            ),
        )


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_chunkreader(db: Database) -> ports.ChunkReader:
        return PgChunks(db)
