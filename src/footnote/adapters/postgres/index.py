# SPDX-License-Identifier: Apache-2.0
"""Chunks, memberships and vectors: incremental indexing, version swap, re-embed, GC.

New chunk and vector rows are invisible to queries until
the swap inserts the memberships of a version that becomes current; GC runs as a
separate job after `ingest.gc_grace`, never inside a swap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from psycopg import sql

from footnote.adapters.postgres.codec import vector_literal
from footnote.adapters.postgres.jobs import insert_job
from footnote.adapters.postgres.session import (
    Conn,
    Database,
    IdFactory,
    execute,
    fetch_all,
    fetch_one,
    new_id,
    require,
)
from footnote.domain.documents import (
    ChunkBatch,
    ChunkId,
    CollectionId,
    DocumentId,
    GcStats,
    JobId,
    Membership,
    SwapResult,
    VersionId,
)
from footnote.domain.embedding import EmbeddingSpace, l2_normalise
from footnote.domain.errors import ConflictError, ValidationError
from footnote.domain.jobs import NewJob
from footnote.domain.tenancy import TenantId

# Chunk `c` (joined to its document `d`) is a member of the document's current version.
CURRENT_MEMBER = sql.SQL(
    "EXISTS (SELECT 1 FROM footnote.version_chunks vc WHERE vc.tenant_id = d.tenant_id "
    "AND vc.version_id = d.current_version_id AND vc.chunk_id = c.id)"
)


class PgIndex:
    """Write side of retrieval; every method runs under the tenant's RLS setting."""

    def __init__(self, db: Database, id_factory: IdFactory = new_id) -> None:
        self._db = db
        self._new_id = id_factory

    def _tf(self, alias: str) -> sql.Composable:
        return self._db.tenant_filter(alias)

    async def existing_chunks(
        self, tenant: TenantId, document_id: DocumentId, hashes: Sequence[str]
    ) -> dict[str, ChunkId]:
        """Chunk ids of the document already stored for these content hashes (diff)."""
        query = sql.SQL(
            "SELECT c.content_hash, c.id FROM footnote.chunks c "
            "WHERE {} AND c.document_id = %(doc)s AND c.content_hash = ANY(%(hashes)s)"
        ).format(self._tf("c"))
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(
                conn, query, {"tenant": tenant, "doc": document_id, "hashes": list(hashes)}
            )
        return {r["content_hash"]: ChunkId(r["id"]) for r in rows}

    async def insert_chunks(self, tenant: TenantId, batch: ChunkBatch) -> dict[str, ChunkId]:
        """Insert chunks (one per distinct hash, existing ones kept); returns hash -> id."""
        unique = list({d.content_hash: d for d in batch.drafts}.values())
        rows = [
            {
                "tenant": tenant,
                "id": self._new_id(),
                "cid": batch.collection_id,
                "doc": batch.document_id,
                "hash": d.content_hash,
                "text": d.text,
                "path": list(d.heading_path),
                "title": batch.title,
                "emb": d.embedded_text,
                "tokens": d.token_count,
                "lang": batch.language.value,
            }
            for d in unique
        ]
        async with self._db.tenant_tx(tenant) as conn:
            if rows:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO footnote.chunks (tenant_id, id, collection_id, document_id, "
                        "content_hash, text, heading_path, title, embedded_text, token_count, "
                        "fts_config) VALUES (%(tenant)s, %(id)s, %(cid)s, %(doc)s, %(hash)s, "
                        "%(text)s, %(path)s, %(title)s, %(emb)s, %(tokens)s, %(lang)s::regconfig) "
                        "ON CONFLICT (tenant_id, document_id, content_hash) DO NOTHING",
                        rows,
                    )
            found = await fetch_all(
                conn,
                "SELECT content_hash, id FROM footnote.chunks WHERE tenant_id = %(tenant)s "
                "AND document_id = %(doc)s AND content_hash = ANY(%(hashes)s)",
                {
                    "tenant": tenant,
                    "doc": batch.document_id,
                    "hashes": [d.content_hash for d in unique],
                },
            )
        return {r["content_hash"]: ChunkId(r["id"]) for r in found}

    async def insert_vectors(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        space: EmbeddingSpace,
        vectors: Sequence[tuple[ChunkId, Sequence[float]]],
    ) -> None:
        """Store L2-normalised vectors of `space` (existing rows kept; pipelined inserts)."""
        rows = []
        for chunk_id, vector in vectors:
            if len(vector) != space.dimensions:
                raise ValidationError(
                    "embedding",
                    f"got {len(vector)} dimensions, space {space.name} has {space.dimensions}",
                    code="embedding_dimension_mismatch",
                )
            rows.append(
                {
                    "tenant": tenant,
                    "chunk": chunk_id,
                    "space": space.id,
                    "cid": collection_id,
                    "vec": vector_literal(l2_normalise(vector)),
                }
            )
        if not rows:
            return
        async with self._db.tenant_tx(tenant) as conn, conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO footnote.chunk_vectors (tenant_id, chunk_id, space_id, "
                "collection_id, embedding) VALUES (%(tenant)s, %(chunk)s, %(space)s, %(cid)s, "
                "%(vec)s::vector) ON CONFLICT (tenant_id, chunk_id, space_id) DO NOTHING",
                rows,
            )

    async def missing_vectors(
        self, tenant: TenantId, collection_id: CollectionId, chunk_ids: Sequence[ChunkId]
    ) -> dict[int, list[ChunkId]]:
        """Per needed space (active, and target while re-embedding): chunks lacking a vector."""
        async with self._db.tenant_tx(tenant) as conn:
            return await self._missing(conn, tenant, collection_id, chunk_ids)

    async def _missing(
        self,
        conn: Conn,
        tenant: TenantId,
        collection_id: CollectionId,
        chunk_ids: Sequence[ChunkId],
    ) -> dict[int, list[ChunkId]]:
        rows = await fetch_all(
            conn,
            sql.SQL(
                "SELECT s.space_id, ids.chunk_id FROM footnote.collections col "
                "CROSS JOIN LATERAL (VALUES (col.active_space_id), (col.target_space_id)) "
                "AS s(space_id) "
                "CROSS JOIN unnest(%(ids)s::uuid[]) AS ids(chunk_id) "
                "WHERE {} AND col.id = %(cid)s AND s.space_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM footnote.chunk_vectors v "
                "WHERE v.tenant_id = col.tenant_id AND v.chunk_id = ids.chunk_id "
                "AND v.space_id = s.space_id) ORDER BY 1, 2"
            ).format(self._tf("col")),
            {"tenant": tenant, "cid": collection_id, "ids": list(chunk_ids)},
        )
        out: dict[int, list[ChunkId]] = {}
        for r in rows:
            out.setdefault(int(r["space_id"]), []).append(ChunkId(r["chunk_id"]))
        return out

    async def current_chunk_count(self, tenant: TenantId, document_id: DocumentId) -> int:
        """Distinct chunks of the document's current version (0 without one)."""
        query = sql.SQL(
            "SELECT count(DISTINCT vc.chunk_id) AS n FROM footnote.documents d "
            "JOIN footnote.version_chunks vc ON vc.tenant_id = d.tenant_id "
            "AND vc.version_id = d.current_version_id WHERE {} AND d.id = %(doc)s"
        ).format(self._tf("d"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "doc": document_id})
        return int(require(row, "count")["n"])

    async def swap(  # noqa: PLR0913 - mirrors the swap transaction's inputs
        self,
        tenant: TenantId,
        document_id: DocumentId,
        version_id: VersionId,
        memberships: Sequence[Membership],
        *,
        jobs: Sequence[NewJob] = (),
        on_current: Sequence[NewJob] = (),
    ) -> SwapResult:
        """Version swap: memberships + current version + chunk_count + jobs, one tx.

        `jobs` are always inserted; `on_current` jobs only when the version became current.

        Locks the document FOR UPDATE, then the collection FOR NO KEY UPDATE
        (FOR SHARE would deadlock two concurrent swaps updating `chunk_count`).
        ConflictError `vectors_missing` when a needed space lacks a vector.
        """
        params = {"tenant": tenant, "doc": document_id, "ver": version_id}
        async with self._db.tenant_tx(tenant) as conn:
            doc = require(
                await fetch_one(
                    conn,
                    sql.SQL(
                        "SELECT d.collection_id, d.current_version_id, cv.version AS current, "
                        "nv.version AS new FROM footnote.documents d "
                        "JOIN footnote.document_versions nv ON nv.tenant_id = d.tenant_id "
                        "AND nv.document_id = d.id AND nv.id = %(ver)s "
                        "LEFT JOIN footnote.document_versions cv ON cv.tenant_id = d.tenant_id "
                        "AND cv.id = d.current_version_id "
                        "WHERE {} AND d.id = %(doc)s FOR UPDATE OF d"
                    ).format(self._tf("d")),
                    params,
                ),
                "document version not found",
            )
            cid = CollectionId(doc["collection_id"])
            await conn.execute(
                "SELECT 1 FROM footnote.collections WHERE tenant_id = %(tenant)s AND id = %(cid)s "
                "FOR NO KEY UPDATE",
                {"tenant": tenant, "cid": cid},
            )
            chunk_ids = list(dict.fromkeys(m.chunk_id for m in memberships))
            missing = await self._missing(conn, tenant, cid, chunk_ids)
            if missing:
                raise ConflictError(
                    f"vectors missing for spaces {sorted(missing)}", code="vectors_missing"
                )
            await self._insert_memberships(conn, tenant, version_id, memberships)
            current = doc["current"] is None or doc["new"] > doc["current"]
            count = await self._flip(conn, params, cid, doc, current=current)
            job_ids = []
            for job in (*jobs, *(on_current if current else ())):
                job_id = JobId(self._new_id())
                await insert_job(conn, tenant, job, job_id)
                job_ids.append(job_id)
        return SwapResult(current, count, tuple(job_ids))

    async def _insert_memberships(
        self,
        conn: Conn,
        tenant: TenantId,
        version_id: VersionId,
        memberships: Sequence[Membership],
    ) -> None:
        if not memberships:
            return
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO footnote.version_chunks (tenant_id, version_id, ordinal, chunk_id, "
                "page_from, page_to) VALUES (%(tenant)s, %(ver)s, %(ord)s, %(chunk)s, %(pf)s, "
                "%(pt)s)",
                [
                    {
                        "tenant": tenant,
                        "ver": version_id,
                        "ord": m.ordinal,
                        "chunk": m.chunk_id,
                        "pf": m.page_from,
                        "pt": m.page_to,
                    }
                    for m in memberships
                    if m.version_id == version_id
                ],
            )

    async def _flip(
        self,
        conn: Conn,
        params: dict[str, Any],
        cid: CollectionId,
        doc: dict[str, Any],
        *,
        current: bool,
    ) -> int:
        if not current:
            await conn.execute(
                "UPDATE footnote.document_versions SET status = 'superseded', updated_at = now() "
                "WHERE tenant_id = %(tenant)s AND id = %(ver)s",
                params,
            )
            row = await fetch_one(
                conn,
                "SELECT chunk_count FROM footnote.collections "
                "WHERE tenant_id = %(tenant)s AND id = %(cid)s",
                {**params, "cid": cid},
            )
            return int(require(row, "collection not found")["chunk_count"])
        flip = {**params, "cid": cid, "old": doc["current_version_id"]}
        await conn.execute(
            "UPDATE footnote.document_versions SET status = CASE WHEN id = %(ver)s THEN 'ready' "
            "ELSE 'superseded' END, updated_at = now() "
            "WHERE tenant_id = %(tenant)s AND id IN (%(ver)s, %(old)s)",
            flip,
        )
        await conn.execute(
            "UPDATE footnote.documents SET current_version_id = %(ver)s, status = 'ready', "
            "error = NULL, updated_at = now() WHERE tenant_id = %(tenant)s AND id = %(doc)s",
            flip,
        )
        row = await fetch_one(
            conn,
            "UPDATE footnote.collections SET chunk_count = GREATEST(chunk_count "
            "+ (SELECT count(DISTINCT chunk_id) FROM footnote.version_chunks "
            "   WHERE tenant_id = %(tenant)s AND version_id = %(ver)s) "
            "- (SELECT count(DISTINCT chunk_id) FROM footnote.version_chunks "
            "   WHERE tenant_id = %(tenant)s AND version_id = %(old)s), 0), updated_at = now() "
            "WHERE tenant_id = %(tenant)s AND id = %(cid)s RETURNING chunk_count",
            flip,
        )
        return int(require(row, "collection not found")["chunk_count"])

    async def gc_document(
        self, tenant: TenantId, document_id: DocumentId, *, delete_superseded_originals: bool
    ) -> GcStats:
        """Remove superseded/failed memberships, unreferenced chunks (and their vectors).

        Chunks are kept while a version of the document is still `processing`, because
        an in-flight ingest has inserted chunks that no membership references yet.
        """
        params = {"tenant": tenant, "doc": document_id, "blobs": delete_superseded_originals}
        async with self._db.tenant_tx(tenant) as conn:
            locked = await fetch_one(
                conn,
                sql.SQL(
                    "SELECT d.id FROM footnote.documents d WHERE {} AND d.id = %(doc)s FOR UPDATE"
                ).format(self._tf("d")),
                params,
            )
            require(locked, "document not found")
            memberships = await execute(
                conn,
                "DELETE FROM footnote.version_chunks vc USING footnote.document_versions v "
                "WHERE v.tenant_id = %(tenant)s AND v.document_id = %(doc)s "
                "AND v.status IN ('superseded', 'failed') "
                "AND vc.tenant_id = v.tenant_id AND vc.version_id = v.id",
                params,
            )
            chunks = await execute(
                conn,
                "DELETE FROM footnote.chunks c WHERE c.tenant_id = %(tenant)s "
                "AND c.document_id = %(doc)s "
                "AND NOT EXISTS (SELECT 1 FROM footnote.version_chunks vc "
                "  WHERE vc.tenant_id = c.tenant_id AND vc.chunk_id = c.id) "
                "AND NOT EXISTS (SELECT 1 FROM footnote.document_versions p "
                "  WHERE p.tenant_id = c.tenant_id AND p.document_id = c.document_id "
                "  AND p.status = 'processing')",
                params,
            )
            blobs = await execute(
                conn,
                "DELETE FROM footnote.document_blobs b USING footnote.document_versions v "
                "WHERE %(blobs)s AND v.tenant_id = %(tenant)s AND v.document_id = %(doc)s "
                "AND v.status = 'superseded' AND b.tenant_id = v.tenant_id "
                "AND b.version_id = v.id",
                params,
            )
        return GcStats(memberships, chunks, blobs)

    async def gc_space(self, tenant: TenantId, collection_id: CollectionId, space_id: int) -> int:
        """Delete a collection's vectors of a space that is neither active nor target."""
        query = sql.SQL(
            "DELETE FROM footnote.chunk_vectors v USING footnote.collections col "
            "WHERE {} AND col.id = %(cid)s AND v.tenant_id = col.tenant_id "
            "AND v.collection_id = col.id AND v.space_id = %(space)s "
            "AND col.active_space_id <> %(space)s "
            "AND col.target_space_id IS DISTINCT FROM %(space)s"
        ).format(self._tf("col"))
        async with self._db.tenant_tx(tenant) as conn:
            return await execute(
                conn, query, {"tenant": tenant, "cid": collection_id, "space": space_id}
            )

    async def reembed_batch(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int, limit: int
    ) -> list[tuple[ChunkId, str]]:
        """Current-version chunks lacking a vector of `space_id` (id, embedded text)."""
        query = sql.SQL(
            "SELECT c.id, c.embedded_text FROM footnote.chunks c "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "WHERE {} AND c.collection_id = %(cid)s AND {} "
            "AND NOT EXISTS (SELECT 1 FROM footnote.chunk_vectors v "
            "WHERE v.tenant_id = c.tenant_id AND v.chunk_id = c.id AND v.space_id = %(space)s) "
            "ORDER BY c.id LIMIT %(limit)s"
        ).format(self._tf("c"), CURRENT_MEMBER)
        params = {"tenant": tenant, "cid": collection_id, "space": space_id, "limit": limit}
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(conn, query, params)
        return [(ChunkId(r["id"]), r["embedded_text"]) for r in rows]

    async def reembed_counts(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int
    ) -> tuple[int, int]:
        """(current-version chunks with a `space_id` vector, all current-version chunks)."""
        query = sql.SQL(
            "SELECT count(*) FILTER (WHERE EXISTS (SELECT 1 FROM footnote.chunk_vectors v "
            "WHERE v.tenant_id = c.tenant_id AND v.chunk_id = c.id AND v.space_id = %(space)s)) "
            "AS done, count(*) AS total FROM footnote.chunks c "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "WHERE {} AND c.collection_id = %(cid)s AND {}"
        ).format(self._tf("c"), CURRENT_MEMBER)
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(
                conn, query, {"tenant": tenant, "cid": collection_id, "space": space_id}
            )
        row = require(row, "counts")
        return int(row["done"]), int(row["total"])

    async def switch_space(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> bool:
        """Re-embed switch: active := target when every current-version chunk has a target vector.

        Returns False (nothing changed) while backfilling is incomplete or no target is set.
        """
        params = {"tenant": tenant, "cid": collection_id}
        async with self._db.tenant_tx(tenant) as conn:
            col = await fetch_one(
                conn,
                sql.SQL(
                    "SELECT col.target_space_id FROM footnote.collections col "
                    "WHERE {} AND col.id = %(cid)s FOR UPDATE"
                ).format(self._tf("col")),
                params,
            )
            target = require(col, "collection not found")["target_space_id"]
            if target is None:
                return False
            lacking = await fetch_one(
                conn,
                sql.SQL(
                    "SELECT 1 FROM footnote.chunks c JOIN footnote.documents d "
                    "ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
                    "WHERE c.tenant_id = %(tenant)s AND c.collection_id = %(cid)s AND {} "
                    "AND NOT EXISTS (SELECT 1 FROM footnote.chunk_vectors v "
                    "WHERE v.tenant_id = c.tenant_id AND v.chunk_id = c.id "
                    "AND v.space_id = %(space)s) LIMIT 1"
                ).format(CURRENT_MEMBER),
                {**params, "space": target},
            )
            if lacking is not None:
                return False
            await conn.execute(
                "UPDATE footnote.collections SET active_space_id = target_space_id, "
                "target_space_id = NULL, reembed = jsonb_set(COALESCE(reembed, '{}'::jsonb), "
                "'{state}', '\"done\"'), updated_at = now() "
                "WHERE tenant_id = %(tenant)s AND id = %(cid)s",
                params,
            )
            for job in jobs:
                await insert_job(conn, tenant, job, JobId(self._new_id()))
        return True


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_indexrepository(db: Database) -> ports.IndexRepository:
        return PgIndex(db)
