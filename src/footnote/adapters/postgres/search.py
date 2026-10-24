# SPDX-License-Identifier: Apache-2.0
"""Search store over Postgres."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from psycopg import sql

from footnote.adapters.postgres.codec import settings_from_json, space_from_row, vector_literal
from footnote.adapters.postgres.index import CURRENT_MEMBER
from footnote.adapters.postgres.session import Conn, Database, fetch_all, fetch_one, jsonb
from footnote.domain.documents import ChunkId, CollectionId, DocumentId
from footnote.domain.embedding import EmbeddingSpace
from footnote.domain.errors import NotFoundError
from footnote.domain.ports import CollectionView, LexicalQuery, SearchFilters, VectorQuery
from footnote.domain.retrieval import (
    Hit,
    LegHit,
    ef_search,
    lexical_rank_sql,
    lexical_tsquery_sql,
)
from footnote.domain.tenancy import TenantId


@dataclass(frozen=True, slots=True)
class SnapshotSettings:
    ef_search: int = 100
    iterative_scan: str = "relaxed_order"
    max_scan_tuples: int = 20_000


_ITERATIVE_SCAN = frozenset({"off", "relaxed_order", "strict_order"})


class PgSearchStore:
    """Opens search snapshots."""

    def __init__(self, db: Database, settings: SnapshotSettings | None = None) -> None:
        self._db = db
        self._settings = settings or SnapshotSettings()
        if self._settings.iterative_scan not in _ITERATIVE_SCAN:
            msg = f"invalid iterative_scan {self._settings.iterative_scan!r}"
            raise ValueError(msg)

    @asynccontextmanager
    async def snapshot(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> AsyncIterator[PgSearchSnapshot]:
        async with self._db.tenant_tx(tenant, read_only_snapshot=True) as conn:
            await conn.execute(
                "SELECT set_config('plan_cache_mode', 'force_custom_plan', true), "
                "set_config('hnsw.iterative_scan', %(scan)s, true), "
                "set_config('hnsw.max_scan_tuples', %(tuples)s, true)",
                {
                    "scan": self._settings.iterative_scan,
                    "tuples": str(int(self._settings.max_scan_tuples)),
                },
            )
            yield PgSearchSnapshot(conn, self._db, tenant, collection_id, self._settings)


class PgSearchSnapshot:
    """Reads one consistent snapshot for a search."""

    def __init__(
        self,
        conn: Conn,
        db: Database,
        tenant: TenantId,
        collection_id: CollectionId,
        settings: SnapshotSettings,
    ) -> None:
        self._conn = conn
        self._db = db
        self._tenant = tenant
        self._collection_id = collection_id
        self._settings = settings
        self._view: CollectionView | None = None

    def _params(self, **extra: Any) -> dict[str, Any]:
        return {"tenant": self._tenant, "cid": self._collection_id, **extra}

    async def collection(self) -> CollectionView:
        if self._view is not None:
            return self._view
        query = sql.SQL(
            "SELECT c.id, c.settings, c.language, c.chunk_count, c.status, "
            "s.id AS space_id, s.name, s.provider, s.model, s.dimensions, s.send_dimensions, "
            "s.normalise, s.calibrated_threshold "
            "FROM footnote.collections c "
            "JOIN footnote.embedding_spaces s ON s.id = c.active_space_id "
            "WHERE {} AND c.id = %(cid)s"
        ).format(self._db.tenant_filter("c"))
        row = await fetch_one(self._conn, query, self._params())
        if row is None or row["status"] != "active":
            raise NotFoundError("collection not found")
        space = space_from_row({**row, "id": row["space_id"]})
        self._view = CollectionView(
            id=CollectionId(row["id"]),
            settings=settings_from_json(row["settings"], row["language"], space.name),
            active_space=space,
            chunk_count=int(row["chunk_count"]),
        )
        return self._view

    def _filters(self, filters: SearchFilters, params: dict[str, Any]) -> sql.Composable:
        parts: list[sql.Composable] = []
        for n, (key, value) in enumerate(sorted(filters.metadata.items())):
            values = value["in"] if isinstance(value, dict) and "in" in value else [value]
            params[f"m{n}"] = [jsonb({key: v}) for v in values]
            parts.append(sql.SQL("d.metadata @> ANY({}::jsonb[])").format(sql.Placeholder(f"m{n}")))
        if filters.document_ids:
            params["doc_ids"] = list(filters.document_ids)
            parts.append(sql.SQL("d.id = ANY(%(doc_ids)s::uuid[])"))
        if filters.created_after is not None:
            params["after"] = filters.created_after
            parts.append(sql.SQL("d.created_at > %(after)s::timestamptz"))
        if filters.created_before is not None:
            params["before"] = filters.created_before
            parts.append(sql.SQL("d.created_at < %(before)s::timestamptz"))
        if not parts:
            return sql.SQL("TRUE")
        return sql.SQL(" AND ").join(parts)

    async def lexical(self, query: LexicalQuery, filters: SearchFilters) -> list[LegHit]:
        """Runs the lexical leg of a search."""
        view = await self.collection()
        params = self._params(query=query.text, limit=query.limit)
        tsquery = sql.SQL(lexical_tsquery_sql(query.match, view.settings.language))
        rank = sql.SQL(lexical_rank_sql(query.rank, "q"))
        stmt = sql.SQL(
            "SELECT c.id, {rank} AS score "
            "FROM (SELECT {tsquery} AS q) AS qt "
            "JOIN footnote.chunks c ON c.tsv @@ qt.q "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "WHERE {tf} AND c.collection_id = %(cid)s AND {current} AND {filters} "
            "ORDER BY score DESC, c.id LIMIT %(limit)s"
        ).format(
            rank=rank,
            tsquery=tsquery,
            tf=self._db.tenant_filter("c"),
            current=CURRENT_MEMBER,
            filters=self._filters(filters, params),
        )
        rows = await fetch_all(self._conn, stmt, params)
        return [LegHit(ChunkId(r["id"]), float(r["score"])) for r in rows]

    async def vector(self, query: VectorQuery, filters: SearchFilters) -> list[LegHit]:
        """Runs the vector leg of a search."""
        stmt, params = await self._vector_statement(query, filters)
        rows = await fetch_all(self._conn, stmt, params)
        return [LegHit(ChunkId(r["chunk_id"]), 1.0 - float(r["dist"])) for r in rows]

    async def explain_vector(self, query: VectorQuery, filters: SearchFilters) -> str:
        stmt, params = await self._vector_statement(query, filters)
        rows = await fetch_all(self._conn, sql.SQL("EXPLAIN {}").format(stmt), params)
        return "\n".join(str(r["QUERY PLAN"]) for r in rows)

    async def _vector_statement(
        self, query: VectorQuery, filters: SearchFilters
    ) -> tuple[sql.Composed, dict[str, Any]]:
        view = await self.collection()
        space = view.active_space
        params = self._params(vec=vector_literal(query.vector), limit=query.limit)
        distance = _distance(space)
        common = sql.SQL(
            "FROM footnote.chunk_vectors v "
            "JOIN footnote.chunks c ON c.tenant_id = v.tenant_id AND c.id = v.chunk_id "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "WHERE {tf} AND v.collection_id = %(cid)s AND v.space_id = {space} "
            "AND {current} AND {filters}"
        ).format(
            tf=self._db.tenant_filter("v"),
            space=sql.Literal(int(space.id)),
            current=CURRENT_MEMBER,
            filters=self._filters(filters, params),
        )
        if query.exact:
            stmt = sql.SQL(
                "SELECT v.chunk_id, {dist} AS dist {common} ORDER BY {dist} + 0, v.chunk_id "
                "LIMIT %(limit)s"
            ).format(dist=distance, common=common)
            return stmt, params
        await self._conn.execute(
            "SELECT set_config('hnsw.ef_search', %(ef)s, true)",
            {"ef": str(ef_search(self._settings.ef_search, query.limit))},
        )
        stmt = sql.SQL(
            "SELECT chunk_id, dist FROM (SELECT v.chunk_id, {dist} AS dist {common} "
            "ORDER BY {dist} LIMIT %(limit)s) AS ann ORDER BY dist, chunk_id"
        ).format(dist=distance, common=common)
        return stmt, params

    async def hits(self, chunk_ids: Sequence[ChunkId]) -> list[Hit]:
        if not chunk_ids:
            return []
        stmt = sql.SQL(
            "SELECT c.id, c.document_id, d.title, c.heading_path, c.text, c.embedded_text, "
            "m.page_from, m.page_to FROM footnote.chunks c "
            "JOIN footnote.documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id "
            "JOIN LATERAL (SELECT vc.page_from, vc.page_to FROM footnote.version_chunks vc "
            "  WHERE vc.tenant_id = d.tenant_id AND vc.version_id = d.current_version_id "
            "  AND vc.chunk_id = c.id ORDER BY vc.ordinal LIMIT 1) AS m ON TRUE "
            "WHERE {tf} AND c.collection_id = %(cid)s AND c.id = ANY(%(ids)s::uuid[])"
        ).format(tf=self._db.tenant_filter("c"))
        rows = await fetch_all(self._conn, stmt, self._params(ids=list(chunk_ids)))
        by_id = {r["id"]: r for r in rows}
        return [_hit(by_id[c]) for c in chunk_ids if c in by_id]

    async def similarities(
        self, chunk_ids: Sequence[ChunkId], vector: Sequence[float]
    ) -> dict[ChunkId, float]:
        if not chunk_ids:
            return {}
        space = (await self.collection()).active_space
        stmt = sql.SQL(
            "SELECT v.chunk_id, {dist} AS dist FROM footnote.chunk_vectors v "
            "WHERE {tf} AND v.collection_id = %(cid)s AND v.space_id = {space} "
            "AND v.chunk_id = ANY(%(ids)s::uuid[])"
        ).format(
            dist=_distance(space),
            tf=self._db.tenant_filter("v"),
            space=sql.Literal(int(space.id)),
        )
        params = self._params(ids=list(chunk_ids), vec=vector_literal(vector))
        rows = await fetch_all(self._conn, stmt, params)
        return {ChunkId(r["chunk_id"]): 1.0 - float(r["dist"]) for r in rows}


def _distance(space: EmbeddingSpace) -> sql.Composable:
    cast = sql.SQL(space.cast())
    return sql.SQL("(v.embedding::{cast} <=> %(vec)s::{cast})").format(cast=cast)


def _hit(row: dict[str, Any]) -> Hit:
    return Hit(
        chunk_id=ChunkId(row["id"]),
        document_id=DocumentId(row["document_id"]),
        title=row["title"],
        heading_path=tuple(row["heading_path"]),
        text=row["text"],
        embedded_text=row["embedded_text"],
        page_from=row["page_from"],
        page_to=row["page_to"],
    )
