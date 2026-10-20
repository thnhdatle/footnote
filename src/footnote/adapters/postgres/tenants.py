# SPDX-License-Identifier: Apache-2.0
"""Tenants and API keys.

`PgTenantAdmin` runs over the admin pool (footnote_admin, `admin_all` policies);
footnote_app has no privileges on `tenants` and `api_keys` and resolves keys only
through the SECURITY DEFINER function `footnote.footnote_resolve_key`.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from footnote.adapters.postgres.codec import quotas_from_json, quotas_to_json
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
from footnote.domain.documents import JobId
from footnote.domain.errors import UnauthenticatedError
from footnote.domain.jobs import NewJob
from footnote.domain.tenancy import (
    ApiKeyRecord,
    IssuedKey,
    Quotas,
    Tenant,
    TenantId,
    TenantStatus,
    TenantUsage,
)

KEY_PREFIX = "fn_"
_KEY_BYTES = 32
_SHOWN_PREFIX_CHARS = 8


def generate_api_key() -> str:
    """`fn_` + base32 of 32 random bytes (lowercase, unpadded)."""
    raw = base64.b32encode(secrets.token_bytes(_KEY_BYTES)).decode().rstrip("=").lower()
    return KEY_PREFIX + raw


def hash_api_key(key: str) -> bytes:
    """SHA-256 of the key as stored in `api_keys.key_hash`."""
    return hashlib.sha256(key.encode()).digest()


def _tenant(row: dict[str, Any]) -> Tenant:
    return Tenant(
        id=TenantId(row["id"]),
        name=row["name"],
        quotas=quotas_from_json(row["quotas"]),
        status=TenantStatus(row["status"]),
    )


def _key(row: dict[str, Any]) -> ApiKeyRecord:
    return ApiKeyRecord(
        id=row["id"],
        tenant_id=TenantId(row["tenant_id"]),
        prefix=row["prefix"],
        name=row["name"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        revoked_at=row["revoked_at"],
        expires_at=row["expires_at"],
    )


class PgTenantAdmin:
    """Tenant and key administration over the admin pool."""

    def __init__(self, admin: Database, id_factory: IdFactory = new_id) -> None:
        self._db = admin
        self._new_id = id_factory

    async def create_tenant(self, name: str, quotas: Quotas) -> Tenant:
        """Insert a tenant."""
        quotas.validate()
        async with self._db.global_tx() as conn:
            row = await fetch_one(
                conn,
                "INSERT INTO footnote.tenants (id, name, quotas) VALUES (%(id)s, %(name)s, %(q)s) "
                "RETURNING id, name, quotas, status",
                {"id": self._new_id(), "name": name, "q": jsonb(quotas_to_json(quotas))},
            )
        return _tenant(require(row, "tenant not created"))

    async def get_tenant(self, tenant: TenantId) -> Tenant:
        """One tenant; NotFoundError when absent."""
        async with self._db.global_tx() as conn:
            row = await fetch_one(
                conn,
                "SELECT id, name, quotas, status FROM footnote.tenants WHERE id = %(id)s",
                {"id": tenant},
            )
        return _tenant(require(row, "tenant not found"))

    async def list_tenants(self, after: TenantId | None, limit: int) -> list[Tenant]:
        """Tenants ordered by id (keyset pagination)."""
        async with self._db.global_tx() as conn:
            rows = await fetch_all(
                conn,
                "SELECT id, name, quotas, status FROM footnote.tenants "
                "WHERE %(after)s::uuid IS NULL OR id > %(after)s ORDER BY id LIMIT %(limit)s",
                {"after": after, "limit": limit},
            )
        return [_tenant(r) for r in rows]

    async def update_tenant(
        self, tenant: TenantId, *, name: str | None = None, quotas: Quotas | None = None
    ) -> Tenant:
        """Change the name and/or quotas."""
        if quotas is not None:
            quotas.validate()
        async with self._db.global_tx() as conn:
            row = await fetch_one(
                conn,
                "UPDATE footnote.tenants SET name = COALESCE(%(name)s, name), "
                "quotas = COALESCE(%(q)s, quotas), updated_at = now() "
                "WHERE id = %(id)s RETURNING id, name, quotas, status",
                {
                    "id": tenant,
                    "name": name,
                    "q": None if quotas is None else jsonb(quotas_to_json(quotas)),
                },
            )
        return _tenant(require(row, "tenant not found"))

    async def mark_deleting(self, tenant: TenantId, jobs: Sequence[NewJob] = ()) -> list[JobId]:
        """Set status `deleting`, revoke every key and enqueue `jobs` (`tenant_delete`), one tx."""
        async with self._db.global_tx() as conn:
            updated = await execute(
                conn,
                "UPDATE footnote.tenants SET status = 'deleting', updated_at = now() "
                "WHERE id = %(id)s",
                {"id": tenant},
            )
            require(updated or None, "tenant not found")
            await execute(
                conn,
                "UPDATE footnote.api_keys SET revoked_at = now() "
                "WHERE tenant_id = %(id)s AND revoked_at IS NULL",
                {"id": tenant},
            )
            ids: list[JobId] = []
            for job in jobs:
                job_id = JobId(self._new_id())
                await insert_job(conn, tenant, job, job_id)
                ids.append(job_id)
            return ids

    async def delete_tenant_data(self, tenant: TenantId) -> None:
        """Remove every row of the tenant (the `tenant_delete` job)."""
        async with self._db.global_tx() as conn:
            await execute(
                conn, "DELETE FROM footnote.jobs WHERE tenant_id = %(id)s", {"id": tenant}
            )
            # Cascades: collections → documents → versions/blobs/memberships/chunks/vectors,
            # webhook deliveries; api_keys and tenant_usage.
            await execute(conn, "DELETE FROM footnote.tenants WHERE id = %(id)s", {"id": tenant})

    async def usage(self, tenant: TenantId) -> TenantUsage:
        """Counts and stored bytes (originals + elements) of a tenant."""
        async with self._db.global_tx() as conn:
            row = await fetch_one(
                conn,
                "SELECT "
                "(SELECT count(*) FROM footnote.collections WHERE tenant_id = %(id)s) AS c, "
                "(SELECT count(*) FROM footnote.documents WHERE tenant_id = %(id)s) AS d, "
                "(SELECT count(*) FROM footnote.chunks WHERE tenant_id = %(id)s) AS ch, "
                "(SELECT count(*) FROM footnote.chunk_vectors WHERE tenant_id = %(id)s) AS v, "
                "(SELECT COALESCE(sum(octet_length(data)), 0) FROM footnote.document_blobs "
                " WHERE tenant_id = %(id)s) "
                "+ (SELECT COALESCE(sum(octet_length(elements)), 0) "
                "   FROM footnote.document_versions WHERE tenant_id = %(id)s) AS b, "
                "EXISTS (SELECT 1 FROM footnote.tenants WHERE id = %(id)s) AS present",
                {"id": tenant},
            )
        row = require(row, "tenant not found")
        require(row["present"] or None, "tenant not found")
        return TenantUsage(
            int(row["c"]), int(row["d"]), int(row["ch"]), int(row["v"]), int(row["b"])
        )

    async def create_key(
        self, tenant: TenantId, name: str, expires_at: datetime | None = None
    ) -> IssuedKey:
        """Issue a key; the plaintext is returned once."""
        key = generate_api_key()
        async with self._db.global_tx() as conn:
            row = await self._insert_key(conn, tenant, name, key, expires_at)
        return IssuedKey(_key(row), key)

    async def _insert_key(
        self, conn: Conn, tenant: TenantId, name: str, key: str, expires_at: datetime | None
    ) -> dict[str, Any]:
        row = await fetch_one(
            conn,
            "INSERT INTO footnote.api_keys (tenant_id, id, key_hash, prefix, name, expires_at) "
            "SELECT t.id, %(id)s, %(hash)s, %(prefix)s, %(name)s, %(exp)s "
            "FROM footnote.tenants t WHERE t.id = %(tenant)s AND t.status = 'active' "
            "RETURNING id, tenant_id, prefix, name, created_at, last_used_at, revoked_at, "
            "expires_at",
            {
                "tenant": tenant,
                "id": self._new_id(),
                "hash": hash_api_key(key),
                "prefix": key[: len(KEY_PREFIX) + _SHOWN_PREFIX_CHARS],
                "name": name,
                "exp": expires_at,
            },
        )
        return require(row, "tenant not found")

    async def list_keys(
        self, tenant: TenantId, after: UUID | None, limit: int
    ) -> list[ApiKeyRecord]:
        """Keys of a tenant (no secrets) ordered by id after `after` (keyset pagination)."""
        async with self._db.global_tx() as conn:
            rows = await fetch_all(
                conn,
                "SELECT id, tenant_id, prefix, name, created_at, last_used_at, revoked_at, "
                "expires_at "
                "FROM footnote.api_keys WHERE tenant_id = %(t)s "
                "AND (%(after)s::uuid IS NULL OR id > %(after)s) ORDER BY id LIMIT %(limit)s",
                {"t": tenant, "after": after, "limit": limit},
            )
        return [_key(r) for r in rows]

    async def revoke_key(self, tenant: TenantId, key_id: UUID) -> None:
        """Revoke a key of this tenant (idempotent)."""
        async with self._db.global_tx() as conn:
            updated = await execute(
                conn,
                "UPDATE footnote.api_keys SET revoked_at = COALESCE(revoked_at, now()) "
                "WHERE tenant_id = %(t)s AND id = %(id)s",
                {"t": tenant, "id": key_id},
            )
        require(updated or None, "key not found")

    async def rotate_key(self, tenant: TenantId, key_id: UUID, grace: timedelta) -> IssuedKey:
        """Issue a replacement; the old key stays valid for `grace` (then expires)."""
        key = generate_api_key()
        async with self._db.global_tx() as conn:
            old = await fetch_one(
                conn,
                "UPDATE footnote.api_keys SET expires_at = LEAST("
                "COALESCE(expires_at, 'infinity'::timestamptz), now() + %(grace)s) "
                "WHERE tenant_id = %(t)s AND id = %(id)s AND revoked_at IS NULL "
                "RETURNING name, expires_at",
                {"t": tenant, "id": key_id, "grace": grace},
            )
            old = require(old, "key not found")
            row = await self._insert_key(conn, tenant, old["name"], key, None)
        return IssuedKey(_key(row), key)


class PgKeyResolver:
    """Resolves tenant API keys as footnote_app via `footnote_resolve_key`."""

    def __init__(self, app: Database) -> None:
        self._db = app

    async def resolve(self, key: str) -> tuple[TenantId, UUID]:
        """(tenant id, key id); UnauthenticatedError for unknown, revoked or expired keys."""
        if not key.startswith(KEY_PREFIX):
            raise UnauthenticatedError("invalid api key")
        async with self._db.global_tx() as conn:
            row = await fetch_one(
                conn,
                "SELECT tenant_id, key_id FROM footnote.footnote_resolve_key(%(h)s)",
                {"h": hash_api_key(key)},
            )
        if row is None:
            raise UnauthenticatedError("invalid api key")
        return TenantId(row["tenant_id"]), row["key_id"]


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_tenantadmin(db: Database) -> ports.TenantAdmin:
        return PgTenantAdmin(db)

    def _conforms_keyresolver(db: Database) -> ports.KeyResolver:
        return PgKeyResolver(db)
