# SPDX-License-Identifier: Apache-2.0
"""Collections repository over the app pool, tenant-scoped by RLS."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from psycopg import sql

from footnote.adapters.postgres.codec import settings_from_json, settings_to_json
from footnote.adapters.postgres.jobs import insert_job
from footnote.adapters.postgres.session import (
    Database,
    IdFactory,
    execute,
    fetch_all,
    fetch_one,
    jsonb,
    new_id,
    require,
)
from footnote.domain.documents import CollectionId, CollectionRecord, JobId, NewCollection
from footnote.domain.jobs import NewJob
from footnote.domain.settings import CollectionSettings, Language
from footnote.domain.tenancy import TenantId

_NOT_FOUND = "collection not found"


_SELECT = sql.SQL(
    "SELECT c.id, c.tenant_id, c.name, c.settings, c.language, c.active_space_id, "
    "c.target_space_id, c.chunk_count, c.reembed, c.webhook_url, c.webhook_secret, c.status, "
    "c.created_at, c.updated_at, COALESCE(t.name, a.name) AS space_name "
    "FROM footnote.collections c "
    "JOIN footnote.embedding_spaces a ON a.id = c.active_space_id "
    "LEFT JOIN footnote.embedding_spaces t ON t.id = c.target_space_id "
)


def _record(row: dict[str, Any]) -> CollectionRecord:
    return CollectionRecord(
        id=CollectionId(row["id"]),
        tenant_id=TenantId(row["tenant_id"]),
        name=row["name"],
        settings=settings_from_json(row["settings"], row["language"], row["space_name"]),
        active_space_id=int(row["active_space_id"]),
        target_space_id=row["target_space_id"],
        chunk_count=int(row["chunk_count"]),
        reembed=row["reembed"],
        webhook_url=row["webhook_url"],
        webhook_secret=row["webhook_secret"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgCollections:
    """CRUD on collections; other tenants' collections look absent."""

    def __init__(self, db: Database, id_factory: IdFactory = new_id) -> None:
        self._db = db
        self._new_id = id_factory

    async def create(self, tenant: TenantId, new: NewCollection) -> CollectionRecord:
        """Insert a collection; ConflictError when the name is taken."""
        cid = self._new_id()
        async with self._db.tenant_tx(tenant) as conn:
            await conn.execute(
                "INSERT INTO footnote.collections (tenant_id, id, name, settings, language, "
                "active_space_id, webhook_url, webhook_secret) VALUES (%(tenant)s, %(id)s, "
                "%(name)s, %(settings)s, %(lang)s, %(space)s, %(url)s, %(secret)s)",
                {
                    "tenant": tenant,
                    "id": cid,
                    "name": new.name,
                    "settings": jsonb(settings_to_json(new.settings)),
                    "lang": new.settings.language.value,
                    "space": new.space_id,
                    "url": new.webhook_url,
                    "secret": new.webhook_secret,
                },
            )
            row = await self._select(conn, tenant, CollectionId(cid), include_deleting=True)
        return _record(require(row, _NOT_FOUND))

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, *, include_deleting: bool = False
    ) -> CollectionRecord:
        """One collection; `deleting` ones are absent unless asked for."""
        async with self._db.tenant_tx(tenant) as conn:
            row = await self._select(conn, tenant, collection_id, include_deleting=include_deleting)
        return _record(require(row, _NOT_FOUND))

    async def _select(
        self,
        conn: Any,
        tenant: TenantId,
        collection_id: CollectionId,
        *,
        include_deleting: bool,
        lock: bool = False,
    ) -> dict[str, Any] | None:
        query = sql.SQL(
            "{} WHERE {} AND c.id = %(id)s AND (%(all)s OR c.status = 'active'){}"
        ).format(
            _SELECT,
            self._db.tenant_filter("c"),
            sql.SQL(" FOR NO KEY UPDATE OF c") if lock else sql.SQL(""),
        )
        return await fetch_one(
            conn, query, {"tenant": tenant, "id": collection_id, "all": include_deleting}
        )

    async def list_page(
        self, tenant: TenantId, after: CollectionId | None, limit: int
    ) -> list[CollectionRecord]:
        """Active collections ordered by id (keyset pagination)."""
        query = sql.SQL(
            "{} WHERE {} AND c.status = 'active' "
            "AND (%(after)s::uuid IS NULL OR c.id > %(after)s) ORDER BY c.id LIMIT %(limit)s"
        ).format(_SELECT, self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(conn, query, {"tenant": tenant, "after": after, "limit": limit})
        return [_record(r) for r in rows]

    async def count(self, tenant: TenantId) -> int:
        """Number of collections (quota `max_collections`)."""
        query = sql.SQL("SELECT count(*) AS n FROM footnote.collections c WHERE {}").format(
            self._db.tenant_filter("c")
        )
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant})
        return int(require(row, "count")["n"])

    async def total_chunks(self, tenant: TenantId) -> int:
        """Sum of `chunk_count` over the tenant's collections (quota `max_chunks`)."""
        query = sql.SQL(
            "SELECT COALESCE(sum(c.chunk_count), 0)::bigint AS n FROM footnote.collections c "
            "WHERE {}"
        ).format(self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant})
        return int(require(row, "total")["n"])

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        settings: CollectionSettings,
        *,
        webhook: tuple[str | None, str | None] | None = None,
        jobs: Sequence[NewJob] = (),
    ) -> tuple[CollectionRecord, list[JobId]]:
        """Replace stored settings (chunking/retrieval/answer), optionally the webhook, and
        enqueue `jobs` in the same transaction.

        `language` and the space change through `set_language` / `set_target_space`
        because they start jobs.
        """
        query = sql.SQL(
            "UPDATE footnote.collections c SET settings = %(settings)s, "
            "webhook_url = CASE WHEN %(set_hook)s THEN %(url)s ELSE webhook_url END, "
            "webhook_secret = CASE WHEN %(set_hook)s THEN %(secret)s ELSE webhook_secret END, "
            "updated_at = now() WHERE {} AND c.id = %(id)s AND c.status = 'active'"
        ).format(self._db.tenant_filter("c"))
        url, secret = webhook if webhook is not None else (None, None)
        async with self._db.tenant_tx(tenant) as conn:
            updated = await execute(
                conn,
                query,
                {
                    "tenant": tenant,
                    "id": collection_id,
                    "settings": jsonb(settings_to_json(settings)),
                    "set_hook": webhook is not None,
                    "url": url,
                    "secret": secret,
                },
            )
            require(updated or None, _NOT_FOUND)
            ids = await self._enqueue(conn, tenant, jobs)
            row = await self._select(conn, tenant, collection_id, include_deleting=False)
        return _record(require(row, _NOT_FOUND)), ids

    async def _enqueue(self, conn: Any, tenant: TenantId, jobs: Sequence[NewJob]) -> list[JobId]:
        ids: list[JobId] = []
        for job in jobs:
            job_id = JobId(self._new_id())
            await insert_job(conn, tenant, job, job_id)
            ids.append(job_id)
        return ids

    async def set_target_space(
        self, tenant: TenantId, collection_id: CollectionId, space_id: int | None
    ) -> CollectionRecord:
        """Start (space id) or cancel (None) a re-embed; progress is reset."""
        query = sql.SQL(
            "UPDATE footnote.collections c SET target_space_id = %(space)s, "
            "reembed = CASE WHEN %(space)s::int IS NULL THEN NULL "
            "ELSE jsonb_build_object('target_space_id', %(space)s::int, 'done', 0, "
            "'total', 0, 'state', 'running') END, updated_at = now() "
            "WHERE {} AND c.id = %(id)s AND c.status = 'active'"
        ).format(self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            updated = await execute(
                conn, query, {"tenant": tenant, "id": collection_id, "space": space_id}
            )
            require(updated or None, _NOT_FOUND)
            row = await self._select(conn, tenant, collection_id, include_deleting=False)
        return _record(require(row, _NOT_FOUND))

    async def set_reembed_progress(
        self, tenant: TenantId, collection_id: CollectionId, done: int, total: int
    ) -> None:
        """Record re-embed progress `{target_space_id, done, total, state}`."""
        query = sql.SQL(
            "UPDATE footnote.collections c SET reembed = jsonb_build_object("
            "'target_space_id', c.target_space_id, 'done', %(done)s::bigint, "
            "'total', %(total)s::bigint, 'state', 'running'), updated_at = now() "
            "WHERE {} AND c.id = %(id)s AND c.target_space_id IS NOT NULL"
        ).format(self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            await execute(
                conn, query, {"tenant": tenant, "id": collection_id, "done": done, "total": total}
            )

    async def mark_deleting(
        self, tenant: TenantId, collection_id: CollectionId, jobs: Sequence[NewJob] = ()
    ) -> list[JobId]:
        """Hide the collection from reads and enqueue `jobs` (its `collection_delete`), one tx."""
        query = sql.SQL(
            "UPDATE footnote.collections c SET status = 'deleting', updated_at = now() "
            "WHERE {} AND c.id = %(id)s AND c.status = 'active'"
        ).format(self._db.tenant_filter("c"))
        async with self._db.tenant_tx(tenant) as conn:
            updated = await execute(conn, query, {"tenant": tenant, "id": collection_id})
            require(updated or None, _NOT_FOUND)
            return await self._enqueue(conn, tenant, jobs)

    async def delete_rows(self, tenant: TenantId, collection_id: CollectionId) -> None:
        """Delete the collection and (by cascade) everything in it."""
        query = sql.SQL("DELETE FROM footnote.collections c WHERE {} AND c.id = %(id)s").format(
            self._db.tenant_filter("c")
        )
        async with self._db.tenant_tx(tenant) as conn:
            await execute(conn, query, {"tenant": tenant, "id": collection_id})

    async def set_language(
        self, tenant: TenantId, collection_id: CollectionId, language: Language
    ) -> None:
        """`fts_reconfig`: switch `language` and every chunk's `fts_config` atomically."""
        params = {"tenant": tenant, "id": collection_id, "lang": language.value}
        async with self._db.tenant_tx(tenant) as conn:
            updated = await execute(
                conn,
                sql.SQL(
                    "UPDATE footnote.collections c SET language = %(lang)s, updated_at = now() "
                    "WHERE {} AND c.id = %(id)s AND c.status = 'active'"
                ).format(self._db.tenant_filter("c")),
                params,
            )
            require(updated or None, _NOT_FOUND)
            await execute(
                conn,
                sql.SQL(
                    "UPDATE footnote.chunks c SET fts_config = %(lang)s::regconfig "
                    "WHERE {} AND c.collection_id = %(id)s "
                    "AND c.fts_config <> %(lang)s::regconfig"
                ).format(self._db.tenant_filter("c")),
                params,
            )


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_collectionrepository(db: Database) -> ports.CollectionRepository:
        return PgCollections(db)
