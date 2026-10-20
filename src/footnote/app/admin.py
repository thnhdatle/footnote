# SPDX-License-Identifier: Apache-2.0
"""Administration use case: tenants, quotas, API keys, export, embedding spaces.

Everything here runs over the admin pool (footnote_admin): footnote_app has no
privileges on `tenants` or `api_keys` and cannot create the per-space HNSW index.
Deleting a tenant revokes its keys and enqueues the `tenant_delete` job in one
transaction (the worker executes it over the admin pool).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from footnote.domain.documents import JobId
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec, index_type_for, validate_space_name
from footnote.domain.errors import ValidationError
from footnote.domain.jobs import JobKind, NewJob
from footnote.domain.ports import Clock, SpaceRegistry, TenantAdmin, TenantExporter
from footnote.domain.tenancy import (
    ApiKeyRecord,
    ExportRecord,
    IssuedKey,
    Quotas,
    Tenant,
    TenantId,
    TenantUsage,
)

NAME_MAX_CHARS = 256
KEY_NAME_MAX_CHARS = 256

type SpaceCompleter = Callable[[SpaceSpec], SpaceSpec]


def _same_spec(spec: SpaceSpec) -> SpaceSpec:
    return spec


@dataclass(frozen=True, slots=True)
class AdminConfig:
    """`tenancy.default_quotas` and `auth.key_rotation_grace`."""

    default_quotas: Quotas = field(default_factory=Quotas)
    key_rotation_grace: timedelta = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class AdminDeps:
    """Admin-pool ports.

    `complete_space` fills what a request may omit from the models manifest
    (dimensions and the calibrated threshold of an `onnx_local` model); the composition
    root provides it, the default passes the spec through unchanged.
    """

    tenants: TenantAdmin
    spaces: SpaceRegistry
    exporter: TenantExporter
    clock: Clock
    complete_space: SpaceCompleter = _same_spec


def _name(field: str, value: str, limit: int) -> str:
    value = value.strip()
    if not 1 <= len(value) <= limit:
        raise ValidationError(field, f"must be 1..{limit} characters")
    return value


class AdminService:
    """Tenant, key and space administration."""

    def __init__(self, deps: AdminDeps, config: AdminConfig | None = None) -> None:
        self._d = deps
        self._cfg = config or AdminConfig()

    @property
    def default_quotas(self) -> Quotas:
        """`tenancy.default_quotas` (base of a create request's partial quotas)."""
        return self._cfg.default_quotas

    async def create_tenant(self, name: str, quotas: Quotas | None = None) -> Tenant:
        """Insert a tenant; quotas default to `tenancy.default_quotas`."""
        quotas = quotas or self._cfg.default_quotas
        quotas.validate()
        return await self._d.tenants.create_tenant(_name("name", name, NAME_MAX_CHARS), quotas)

    async def list_tenants(self, after: TenantId | None, limit: int) -> list[Tenant]:
        """Tenants ordered by id."""
        return await self._d.tenants.list_tenants(after, limit)

    async def get_tenant(self, tenant: TenantId) -> Tenant:
        """One tenant (404 when absent)."""
        return await self._d.tenants.get_tenant(tenant)

    async def update_tenant(
        self, tenant: TenantId, *, name: str | None = None, quotas: Quotas | None = None
    ) -> Tenant:
        """Change the name and/or quotas."""
        if quotas is not None:
            quotas.validate()
        if name is not None:
            name = _name("name", name, NAME_MAX_CHARS)
        return await self._d.tenants.update_tenant(tenant, name=name, quotas=quotas)

    async def delete_tenant(self, tenant: TenantId) -> JobId:
        """Revoke every key now; data, keys and the tenant row go with `tenant_delete`."""
        ids = await self._d.tenants.mark_deleting(
            tenant, (NewJob(JobKind.TENANT_DELETE.value, {"tenant_id": str(tenant)}),)
        )
        return ids[0]

    async def usage(self, tenant: TenantId) -> TenantUsage:
        """Collections, documents, chunks, vectors and stored bytes of a tenant."""
        return await self._d.tenants.usage(tenant)

    async def export(
        self, tenant: TenantId, *, include_vectors: bool = False, include_originals: bool = False
    ) -> AsyncIterator[ExportRecord]:
        """Stream the tenant's data; 404 is raised before the first record."""
        await self._d.tenants.get_tenant(tenant)
        return self._d.exporter.export(
            tenant, include_vectors=include_vectors, include_originals=include_originals
        )

    async def create_key(
        self, tenant: TenantId, name: str, expires_at: datetime | None = None
    ) -> IssuedKey:
        """Issue a key (plaintext returned once); `expires_at` must lie in the future."""
        if expires_at is not None:
            if expires_at.tzinfo is None:
                raise ValidationError("expires_at", "must include a time zone")
            now = datetime.fromtimestamp(self._d.clock.now(), UTC)
            if expires_at <= now:
                raise ValidationError("expires_at", "must be in the future")
        return await self._d.tenants.create_key(
            tenant, _name("name", name, KEY_NAME_MAX_CHARS), expires_at
        )

    async def list_keys(
        self, tenant: TenantId, after: UUID | None, limit: int
    ) -> list[ApiKeyRecord]:
        """Keys of a tenant without secrets, ordered by id (404 for an unknown tenant)."""
        await self._d.tenants.get_tenant(tenant)
        return await self._d.tenants.list_keys(tenant, after, limit)

    async def revoke_key(self, tenant: TenantId, key_id: UUID) -> None:
        """Revoke a key now (idempotent)."""
        await self._d.tenants.revoke_key(tenant, key_id)

    async def rotate_key(self, tenant: TenantId, key_id: UUID) -> IssuedKey:
        """Issue a replacement; the old key stays valid for `auth.key_rotation_grace`."""
        return await self._d.tenants.rotate_key(tenant, key_id, self._cfg.key_rotation_grace)

    async def register_space(self, spec: SpaceSpec) -> EmbeddingSpace:
        """Register a space (idempotent by name; 409 `space_limit_reached`)."""
        validate_space_name(spec.name)
        spec = self._d.complete_space(spec)
        index_type_for(spec.dimensions)
        if spec.calibrated_threshold is not None and not 0.0 <= spec.calibrated_threshold <= 1.0:
            raise ValidationError("calibrated_threshold", "must be in [0, 1] or null")
        return await self._d.spaces.register(spec)

    async def list_spaces(self, after: int | None, limit: int) -> list[EmbeddingSpace]:
        """Spaces ordered by id (keyset pagination)."""
        return await self._d.spaces.list_spaces_page(after, limit)

    async def all_spaces(self) -> list[EmbeddingSpace]:
        """Every registered space."""
        return await self._d.spaces.list_spaces()

    async def delete_space(self, name: str) -> None:
        """Drop a space and its index; 409 `space_in_use` while a collection uses it."""
        await self._d.spaces.delete(name)
