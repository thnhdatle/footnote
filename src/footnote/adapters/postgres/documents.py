# SPDX-License-Identifier: Apache-2.0
"""Documents, versions and stored originals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psycopg import sql

from footnote.adapters.postgres.codec import compress, decode_elements, decompress, encode_elements
from footnote.adapters.postgres.jobs import insert_job
from footnote.adapters.postgres.session import (
    Conn,
    Database,
    IdFactory,
    execute,
    fetch_all,
    fetch_one,
    jsonb,
    new_id,
    require,
)
from footnote.domain.documents import (
    CollectionId,
    DocumentId,
    DocumentRecord,
    DocumentStatus,
    Element,
    JobId,
    NewUpload,
    UploadResult,
    VersionId,
    VersionRecord,
    VersionStatus,
)
from footnote.domain.errors import NotFoundError
from footnote.domain.jobs import NewJob
from footnote.domain.tenancy import TenantId

_DOC_NOT_FOUND = "document not found"
_VERSION_NOT_FOUND = "document version not found"


_NO_NEWER_VERSION = sql.SQL(
    " AND NOT EXISTS (SELECT 1 FROM footnote.document_versions n "
    "JOIN footnote.document_versions v ON v.tenant_id = n.tenant_id AND v.id = %(ver)s "
    "WHERE n.tenant_id = d.tenant_id AND n.document_id = d.id AND n.version > v.version)"
)
_DOC_SELECT = sql.SQL(
    "SELECT d.id, d.collection_id, d.external_id, d.title, d.metadata, d.status, d.error, "
    "d.current_version_id, cv.version AS current_version, d.created_at, d.updated_at "
    "FROM footnote.documents d "
    "JOIN footnote.collections c ON c.tenant_id = d.tenant_id AND c.id = d.collection_id "
    "AND c.status = 'active' "
    "LEFT JOIN footnote.document_versions cv "
    "ON cv.tenant_id = d.tenant_id AND cv.id = d.current_version_id "
)
_VERSION_SELECT = sql.SQL(
    "SELECT v.id, v.document_id, d.collection_id, v.version, v.source_sha256, v.media_type, "
    "v.bytes, v.pages, v.indexed_title, v.chunking_hash, v.status, v.error, "
    "EXISTS (SELECT 1 FROM footnote.document_blobs b "
    "        WHERE b.tenant_id = v.tenant_id AND b.version_id = v.id) AS has_original, "
    "v.created_at, v.updated_at "
    "FROM footnote.document_versions v "
    "JOIN footnote.documents d ON d.tenant_id = v.tenant_id AND d.id = v.document_id "
)


def _doc(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        id=DocumentId(row["id"]),
        collection_id=CollectionId(row["collection_id"]),
        external_id=row["external_id"],
        title=row["title"],
        metadata=row["metadata"],
        status=DocumentStatus(row["status"]),
        error=row["error"],
        current_version_id=row["current_version_id"],
        current_version=row["current_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _version(row: dict[str, Any]) -> VersionRecord:
    return VersionRecord(
        id=VersionId(row["id"]),
        document_id=DocumentId(row["document_id"]),
        collection_id=CollectionId(row["collection_id"]),
        version=int(row["version"]),
        source_sha256=bytes(row["source_sha256"]),
        media_type=row["media_type"],
        bytes=int(row["bytes"]),
        pages=row["pages"],
        indexed_title=row["indexed_title"],
        chunking_hash=row["chunking_hash"],
        status=VersionStatus(row["status"]),
        error=row["error"],
        has_original=bool(row["has_original"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgDocuments:
    """Stores and reads documents and their versions."""

    def __init__(self, db: Database, id_factory: IdFactory = new_id) -> None:
        self._db = db
        self._new_id = id_factory

    def _tf(self, alias: str) -> sql.Composable:
        return self._db.tenant_filter(alias)

    async def upload(self, tenant: TenantId, up: NewUpload) -> UploadResult:
        """Creates or versions a document and enqueues its ingest job."""
        async with self._db.tenant_tx(tenant) as conn:
            await self._lock_active_collection(conn, tenant, up.collection_id)
            doc_id = DocumentId(self._new_id())
            row = require(
                await fetch_one(
                    conn,
                    "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id, "
                    "title, metadata) VALUES (%(tenant)s, %(id)s, %(cid)s, %(ext)s, %(title)s, "
                    "%(meta)s) ON CONFLICT (tenant_id, collection_id, external_id) "
                    "DO UPDATE SET updated_at = now() RETURNING id",
                    {
                        "tenant": tenant,
                        "id": doc_id,
                        "cid": up.collection_id,
                        "ext": up.external_id or str(doc_id),
                        "title": up.title,
                        "meta": jsonb(up.metadata),
                    },
                ),
                _DOC_NOT_FOUND,
            )
            doc_id = DocumentId(row["id"])
            if await self._is_unchanged(conn, tenant, doc_id, up):
                await conn.execute(
                    "UPDATE footnote.documents SET metadata = %(meta)s, updated_at = now() "
                    "WHERE tenant_id = %(tenant)s AND id = %(id)s",
                    {"tenant": tenant, "id": doc_id, "meta": jsonb(up.metadata)},
                )
                return UploadResult(doc_id, unchanged=True)
            return await self._new_version(conn, tenant, doc_id, up)

    async def _lock_active_collection(
        self, conn: Conn, tenant: TenantId, collection_id: CollectionId
    ) -> None:
        query = sql.SQL(
            "SELECT 1 FROM footnote.collections c WHERE {} AND c.id = %(cid)s "
            "AND c.status = 'active' FOR KEY SHARE"
        ).format(self._tf("c"))
        row = await fetch_one(conn, query, {"tenant": tenant, "cid": collection_id})
        require(row, "collection not found")

    async def _is_unchanged(
        self, conn: Conn, tenant: TenantId, doc_id: DocumentId, up: NewUpload
    ) -> bool:
        row = await fetch_one(
            conn,
            "SELECT v.source_sha256, v.indexed_title, v.chunking_hash "
            "FROM footnote.documents d JOIN footnote.document_versions v "
            "ON v.tenant_id = d.tenant_id AND v.id = d.current_version_id "
            "WHERE d.tenant_id = %(tenant)s AND d.id = %(id)s",
            {"tenant": tenant, "id": doc_id},
        )
        return row is not None and (
            bytes(row["source_sha256"]) == up.source_sha256
            and row["indexed_title"] == up.title
            and row["chunking_hash"] == up.chunking_hash
        )

    async def _new_version(
        self, conn: Conn, tenant: TenantId, doc_id: DocumentId, up: NewUpload
    ) -> UploadResult:
        version_id = VersionId(self._new_id())
        job_id = JobId(self._new_id())
        row = require(
            await fetch_one(
                conn,
                "INSERT INTO footnote.document_versions (tenant_id, id, document_id, version, "
                "source_sha256, media_type, bytes, pages, indexed_title, chunking_hash) "
                "SELECT %(tenant)s, %(vid)s, %(did)s, COALESCE(max(version), 0) + 1, %(sha)s, "
                "%(mt)s, %(size)s, %(pages)s, %(title)s, %(chash)s "
                "FROM footnote.document_versions "
                "WHERE tenant_id = %(tenant)s AND document_id = %(did)s RETURNING version",
                {
                    "tenant": tenant,
                    "vid": version_id,
                    "did": doc_id,
                    "sha": up.source_sha256,
                    "mt": up.media_type,
                    "size": len(up.data),
                    "pages": up.pages,
                    "title": up.title,
                    "chash": up.chunking_hash,
                },
            ),
            _VERSION_NOT_FOUND,
        )
        await conn.execute(
            "INSERT INTO footnote.document_blobs (tenant_id, version_id, data) "
            "VALUES (%(tenant)s, %(vid)s, %(data)s)",
            {"tenant": tenant, "vid": version_id, "data": compress(up.data)},
        )
        await conn.execute(
            "UPDATE footnote.documents SET title = %(title)s, metadata = %(meta)s, "
            "status = 'queued', error = NULL, updated_at = now() "
            "WHERE tenant_id = %(tenant)s AND id = %(id)s",
            {"tenant": tenant, "id": doc_id, "title": up.title, "meta": jsonb(up.metadata)},
        )
        args = {
            "collection_id": str(up.collection_id),
            "document_id": str(doc_id),
            "version_id": str(version_id),
        }
        await insert_job(conn, tenant, NewJob("ingest", args, up.ingest_delay), job_id)
        usage = require(
            await fetch_one(
                conn,
                "INSERT INTO footnote.tenant_usage (tenant_id, day, upload_bytes) "
                "VALUES (%(tenant)s, current_date, %(size)s) ON CONFLICT (tenant_id, day) "
                "DO UPDATE SET upload_bytes = footnote.tenant_usage.upload_bytes + %(size)s "
                "RETURNING upload_bytes",
                {"tenant": tenant, "size": len(up.data)},
            ),
            "usage",
        )
        return UploadResult(
            doc_id,
            unchanged=False,
            version_id=version_id,
            version=int(row["version"]),
            job_id=job_id,
            upload_bytes_today=int(usage["upload_bytes"]),
        )

    async def reversion(
        self, tenant: TenantId, document_id: DocumentId, chunking_hash: str
    ) -> UploadResult | None:
        """Creates a new version from the stored elements of the current one."""
        params = {"tenant": tenant, "id": document_id}
        async with self._db.tenant_tx(tenant) as conn:
            doc = require(
                await fetch_one(
                    conn,
                    sql.SQL(
                        "SELECT d.collection_id, d.current_version_id FROM footnote.documents d "
                        "JOIN footnote.collections c ON c.tenant_id = d.tenant_id "
                        "AND c.id = d.collection_id AND c.status = 'active' "
                        "WHERE {} AND d.id = %(id)s FOR UPDATE OF d"
                    ).format(self._tf("d")),
                    params,
                ),
                _DOC_NOT_FOUND,
            )
            current = doc["current_version_id"]
            if current is None:
                return None
            latest = require(
                await fetch_one(
                    conn,
                    "SELECT chunking_hash, status FROM footnote.document_versions "
                    "WHERE tenant_id = %(tenant)s AND document_id = %(id)s "
                    "ORDER BY version DESC LIMIT 1",
                    params,
                ),
                _VERSION_NOT_FOUND,
            )
            if latest["status"] == "processing" or (
                latest["status"] == "ready" and latest["chunking_hash"] == chunking_hash
            ):
                return None
            version_id = VersionId(self._new_id())
            job_id = JobId(self._new_id())
            row = await fetch_one(
                conn,
                "INSERT INTO footnote.document_versions (tenant_id, id, document_id, version, "
                "source_sha256, media_type, bytes, pages, indexed_title, elements, "
                "chunking_hash) "
                "SELECT cur.tenant_id, %(new)s, cur.document_id, "
                "(SELECT max(m.version) + 1 FROM footnote.document_versions m "
                " WHERE m.tenant_id = cur.tenant_id AND m.document_id = cur.document_id), "
                "cur.source_sha256, cur.media_type, cur.bytes, cur.pages, cur.indexed_title, "
                "cur.elements, %(chash)s FROM footnote.document_versions cur "
                "WHERE cur.tenant_id = %(tenant)s AND cur.id = %(cur)s "
                "AND cur.elements IS NOT NULL RETURNING version",
                {**params, "new": version_id, "cur": current, "chash": chunking_hash},
            )
            if row is None:
                return None
            await conn.execute(
                "INSERT INTO footnote.document_blobs (tenant_id, version_id, data) "
                "SELECT tenant_id, %(new)s, data FROM footnote.document_blobs "
                "WHERE tenant_id = %(tenant)s AND version_id = %(cur)s",
                {**params, "new": version_id, "cur": current},
            )
            await conn.execute(
                "UPDATE footnote.documents SET status = 'queued', error = NULL, "
                "updated_at = now() WHERE tenant_id = %(tenant)s AND id = %(id)s",
                params,
            )
            args = {
                "collection_id": str(doc["collection_id"]),
                "document_id": str(document_id),
                "version_id": str(version_id),
            }
            await insert_job(conn, tenant, NewJob("ingest", args), job_id)
        return UploadResult(
            document_id,
            unchanged=False,
            version_id=version_id,
            version=int(row["version"]),
            job_id=job_id,
        )

    async def uploaded_bytes_today(self, tenant: TenantId) -> int:
        query = sql.SQL(
            "SELECT COALESCE(sum(u.upload_bytes), 0) AS n FROM footnote.tenant_usage u "
            "WHERE {} AND u.day = current_date"
        ).format(self._tf("u"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant})
        return int(require(row, "usage")["n"])

    async def count(self, tenant: TenantId) -> int:
        query = sql.SQL("SELECT count(*) AS n FROM footnote.documents d WHERE {}").format(
            self._tf("d")
        )
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant})
        return int(require(row, "count")["n"])

    async def get(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> DocumentRecord:
        query = sql.SQL("{} WHERE {} AND d.collection_id = %(cid)s AND d.id = %(id)s").format(
            _DOC_SELECT, self._tf("d")
        )
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(
                conn, query, {"tenant": tenant, "cid": collection_id, "id": document_id}
            )
        return _doc(require(row, _DOC_NOT_FOUND))

    async def find(
        self, tenant: TenantId, collection_id: CollectionId, external_id: str
    ) -> DocumentRecord | None:
        query = sql.SQL(
            "{} WHERE {} AND d.collection_id = %(cid)s AND d.external_id = %(ext)s"
        ).format(_DOC_SELECT, self._tf("d"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(
                conn, query, {"tenant": tenant, "cid": collection_id, "ext": external_id}
            )
        return None if row is None else _doc(row)

    async def list_page(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        after: DocumentId | None,
        limit: int,
    ) -> list[DocumentRecord]:
        query = sql.SQL(
            "{} WHERE {} AND d.collection_id = %(cid)s "
            "AND (%(after)s::uuid IS NULL OR d.id > %(after)s) ORDER BY d.id LIMIT %(limit)s"
        ).format(_DOC_SELECT, self._tf("d"))
        params = {"tenant": tenant, "cid": collection_id, "after": after, "limit": limit}
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(conn, query, params)
        return [_doc(r) for r in rows]

    async def versions(self, tenant: TenantId, document_id: DocumentId) -> list[VersionRecord]:
        query = sql.SQL("{} WHERE {} AND v.document_id = %(id)s ORDER BY v.version").format(
            _VERSION_SELECT, self._tf("v")
        )
        async with self._db.tenant_tx(tenant) as conn:
            rows = await fetch_all(conn, query, {"tenant": tenant, "id": document_id})
        return [_version(r) for r in rows]

    async def version(self, tenant: TenantId, version_id: VersionId) -> VersionRecord:
        query = sql.SQL("{} WHERE {} AND v.id = %(id)s").format(_VERSION_SELECT, self._tf("v"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "id": version_id})
        return _version(require(row, _VERSION_NOT_FOUND))

    async def update(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> DocumentRecord:
        query = sql.SQL(
            "UPDATE footnote.documents d SET metadata = COALESCE(%(meta)s, d.metadata), "
            "title = COALESCE(%(title)s, d.title), updated_at = now() "
            "WHERE {} AND d.collection_id = %(cid)s AND d.id = %(id)s"
        ).format(self._tf("d"))
        params = {
            "tenant": tenant,
            "cid": collection_id,
            "id": document_id,
            "meta": None if metadata is None else jsonb(metadata),
            "title": title,
        }
        async with self._db.tenant_tx(tenant) as conn:
            require(await execute(conn, query, params) or None, _DOC_NOT_FOUND)
        return await self.get(tenant, collection_id, document_id)

    async def set_status(
        self,
        tenant: TenantId,
        document_id: DocumentId,
        status: DocumentStatus,
        error: dict[str, Any] | None = None,
        *,
        version_id: VersionId | None = None,
    ) -> None:
        guard = _NO_NEWER_VERSION if version_id is not None else sql.SQL("")
        query = sql.SQL(
            "UPDATE footnote.documents d SET status = %(status)s, error = %(error)s, "
            "updated_at = now() WHERE {} AND d.id = %(id)s{}"
        ).format(self._tf("d"), guard)
        params = {
            "tenant": tenant,
            "id": document_id,
            "ver": version_id,
            "status": status.value,
            "error": None if error is None else jsonb(error),
        }
        async with self._db.tenant_tx(tenant) as conn:
            updated = await execute(conn, query, params)
        if version_id is None:
            require(updated or None, _DOC_NOT_FOUND)

    async def store_elements(
        self,
        tenant: TenantId,
        version_id: VersionId,
        elements: list[Element],
        pages: int | None = None,
    ) -> None:
        query = sql.SQL(
            "UPDATE footnote.document_versions v SET elements = %(el)s, "
            "pages = COALESCE(%(pages)s, v.pages), updated_at = now() WHERE {} AND v.id = %(id)s"
        ).format(self._tf("v"))
        params = {
            "tenant": tenant,
            "id": version_id,
            "el": encode_elements(elements),
            "pages": pages,
        }
        async with self._db.tenant_tx(tenant) as conn:
            require(await execute(conn, query, params) or None, _VERSION_NOT_FOUND)

    async def elements(self, tenant: TenantId, version_id: VersionId) -> list[Element]:
        query = sql.SQL(
            "SELECT v.elements FROM footnote.document_versions v WHERE {} AND v.id = %(id)s"
        ).format(self._tf("v"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "id": version_id})
        data = require(row, _VERSION_NOT_FOUND)["elements"]
        if data is None:
            raise NotFoundError("document version has no parsed elements")
        return decode_elements(bytes(data))

    async def fail_version(
        self, tenant: TenantId, version_id: VersionId, error: dict[str, Any]
    ) -> None:
        params = {"tenant": tenant, "id": version_id, "error": jsonb(error)}
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(
                conn,
                sql.SQL(
                    "UPDATE footnote.document_versions v SET status = 'failed', "
                    "error = %(error)s, updated_at = now() WHERE {} AND v.id = %(id)s "
                    "RETURNING v.document_id"
                ).format(self._tf("v")),
                params,
            )
            doc = require(row, _VERSION_NOT_FOUND)["document_id"]
            await conn.execute(
                "UPDATE footnote.documents d SET status = 'failed', error = %(error)s, "
                "updated_at = now() WHERE d.tenant_id = %(tenant)s AND d.id = %(doc)s "
                "AND NOT EXISTS (SELECT 1 FROM footnote.document_versions n "
                "JOIN footnote.document_versions v ON v.tenant_id = n.tenant_id "
                "AND v.id = %(id)s WHERE n.tenant_id = d.tenant_id AND n.document_id = d.id "
                "AND n.version > v.version)",
                {**params, "doc": doc},
            )

    async def original(
        self,
        tenant: TenantId,
        collection_id: CollectionId,
        document_id: DocumentId,
        version: int | None = None,
    ) -> tuple[str, bytes]:
        query = sql.SQL(
            "SELECT v.media_type, b.data FROM footnote.documents d "
            "JOIN footnote.collections c ON c.tenant_id = d.tenant_id AND c.id = d.collection_id "
            "AND c.status = 'active' "
            "JOIN footnote.document_versions v ON v.tenant_id = d.tenant_id "
            "AND v.document_id = d.id "
            "AND ((%(ver)s::int IS NULL AND v.id = d.current_version_id) OR v.version = %(ver)s) "
            "LEFT JOIN footnote.document_blobs b ON b.tenant_id = v.tenant_id "
            "AND b.version_id = v.id "
            "WHERE {} AND d.collection_id = %(cid)s AND d.id = %(id)s"
        ).format(self._tf("d"))
        params = {"tenant": tenant, "cid": collection_id, "id": document_id, "ver": version}
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, params)
        row = require(row, _VERSION_NOT_FOUND)
        if row["data"] is None:
            raise NotFoundError("original not retained", code="original_not_retained")
        return row["media_type"], decompress(bytes(row["data"]))

    async def source(self, tenant: TenantId, version_id: VersionId) -> bytes:
        query = sql.SQL(
            "SELECT b.data FROM footnote.document_blobs b WHERE {} AND b.version_id = %(id)s"
        ).format(self._tf("b"))
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(conn, query, {"tenant": tenant, "id": version_id})
        if row is None:
            raise NotFoundError("original not retained", code="original_not_retained")
        return decompress(bytes(row["data"]))

    async def delete_original(self, tenant: TenantId, version_id: VersionId) -> None:
        query = sql.SQL(
            "DELETE FROM footnote.document_blobs b WHERE {} AND b.version_id = %(id)s"
        ).format(self._tf("b"))
        async with self._db.tenant_tx(tenant) as conn:
            await execute(conn, query, {"tenant": tenant, "id": version_id})

    async def delete(
        self, tenant: TenantId, collection_id: CollectionId, document_id: DocumentId
    ) -> None:
        """Deletes a document with its versions and chunks."""
        params = {"tenant": tenant, "cid": collection_id, "id": document_id}
        async with self._db.tenant_tx(tenant) as conn:
            row = await fetch_one(
                conn,
                sql.SQL(
                    "SELECT d.current_version_id FROM footnote.documents d "
                    "WHERE {} AND d.collection_id = %(cid)s AND d.id = %(id)s FOR UPDATE"
                ).format(self._tf("d")),
                params,
            )
            current = require(row, _DOC_NOT_FOUND)["current_version_id"]
            await conn.execute(
                "UPDATE footnote.collections SET chunk_count = GREATEST(chunk_count - ("
                "SELECT count(DISTINCT chunk_id) FROM footnote.version_chunks "
                "WHERE tenant_id = %(tenant)s AND version_id = %(cur)s), 0), updated_at = now() "
                "WHERE tenant_id = %(tenant)s AND id = %(cid)s",
                {**params, "cur": current},
            )
            await conn.execute(
                "DELETE FROM footnote.documents WHERE tenant_id = %(tenant)s AND id = %(id)s",
                params,
            )


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_documentrepository(db: Database) -> ports.DocumentRepository:
        return PgDocuments(db)

    def _conforms_blobstore(db: Database) -> ports.BlobStore:
        return PgDocuments(db)
