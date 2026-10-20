# SPDX-License-Identifier: Apache-2.0
"""Fakes of the admin-pool and edge ports the API needs, over the app-layer FakeDb."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

from footnote.domain.deliveries import DeliveryRecord
from footnote.domain.documents import (
    ChunkDetail,
    ChunkId,
    ChunkPlacement,
    CollectionId,
    InspectedFile,
    JobId,
)
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec
from footnote.domain.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    UnauthenticatedError,
)
from footnote.domain.jobs import NewJob
from footnote.domain.ports import SearchSnapshot
from footnote.domain.tenancy import (
    ApiKeyRecord,
    ExportRecord,
    IssuedKey,
    Quotas,
    Tenant,
    TenantId,
    TenantStatus,
    TenantUsage,
)
from tests.unit.app.fakes import EPOCH, FakeDb, FakeSearchStore, FakeSpaces, FakeTenants


class FakeKeys:
    """KeyResolver over an in-memory key table (`revoked`/`expired` keys are unknown)."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.keys: dict[str, ApiKeyRecord] = {}
        self.calls: list[str] = []

    def issue(self, tenant: TenantId, name: str = "default") -> IssuedKey:
        key = f"fn_test{len(self.keys):04d}{tenant.hex[:8]}"
        record = ApiKeyRecord(
            UUID(int=self.db.new_id().int), tenant, key[:11], name, EPOCH, None, None, None
        )
        self.keys[key] = record
        return IssuedKey(record, key)

    def revoke(self, key: str) -> None:
        self.keys[key] = replace(self.keys[key], revoked_at=EPOCH)

    async def resolve(self, key: str) -> tuple[TenantId, UUID]:
        self.calls.append(key)
        record = self.keys.get(key)
        if record is None or record.revoked_at is not None:
            raise UnauthenticatedError("invalid api key")
        if record.expires_at is not None and record.expires_at <= datetime.now(UTC):
            raise UnauthenticatedError("invalid api key")
        return record.tenant_id, record.id


class FakeTenantAdmin(FakeTenants):
    """TenantAdmin over FakeDb and FakeKeys."""

    def __init__(self, db: FakeDb, keys: FakeKeys) -> None:
        super().__init__(db)
        self.keys = keys

    async def create_tenant(self, name: str, quotas: Quotas) -> Tenant:
        tenant = TenantId(self.db.new_id())
        self.db.tenants[tenant] = Tenant(tenant, name, quotas)
        return self.db.tenants[tenant]

    async def list_tenants(self, after: TenantId | None, limit: int) -> list[Tenant]:
        ordered = sorted(self.db.tenants.values(), key=lambda t: t.id)
        return [t for t in ordered if after is None or t.id > after][:limit]

    async def update_tenant(
        self, tenant: TenantId, *, name: str | None = None, quotas: Quotas | None = None
    ) -> Tenant:
        current = await self.get_tenant(tenant)
        updated = replace(
            current,
            name=name if name is not None else current.name,
            quotas=quotas if quotas is not None else current.quotas,
        )
        self.db.tenants[tenant] = updated
        return updated

    async def mark_deleting(self, tenant: TenantId, jobs: Sequence[NewJob] = ()) -> list[JobId]:
        current = await self.get_tenant(tenant)
        self.db.tenants[tenant] = replace(current, status=TenantStatus.DELETING)
        for key, record in list(self.keys.keys.items()):
            if record.tenant_id == tenant:
                self.keys.revoke(key)
        return [self.db.insert_job(tenant, j) for j in jobs]

    async def usage(self, tenant: TenantId) -> TenantUsage:
        await self.get_tenant(tenant)
        count = sum(1 for (t, _) in self.db.cols if t == tenant)
        docs = sum(1 for (t, _) in self.db.docs if t == tenant)
        return TenantUsage(count, docs, 0, 0, 0)

    async def create_key(
        self, tenant: TenantId, name: str, expires_at: datetime | None = None
    ) -> IssuedKey:
        await self.get_tenant(tenant)
        issued = self.keys.issue(tenant, name)
        if expires_at is not None:
            record = replace(issued.record, expires_at=expires_at)
            self.keys.keys[issued.key] = record
            issued = IssuedKey(record, issued.key)
        return issued

    async def list_keys(
        self, tenant: TenantId, after: UUID | None, limit: int
    ) -> list[ApiKeyRecord]:
        mine = sorted(
            (r for r in self.keys.keys.values() if r.tenant_id == tenant), key=lambda r: r.id
        )
        return [r for r in mine if after is None or r.id > after][:limit]

    def _key(self, tenant: TenantId, key_id: UUID) -> str:
        for key, record in self.keys.keys.items():
            if record.tenant_id == tenant and record.id == key_id:
                return key
        raise NotFoundError("key not found")

    async def revoke_key(self, tenant: TenantId, key_id: UUID) -> None:
        self.keys.revoke(self._key(tenant, key_id))

    async def rotate_key(self, tenant: TenantId, key_id: UUID, grace: timedelta) -> IssuedKey:
        old = self._key(tenant, key_id)
        record = self.keys.keys[old]
        self.keys.keys[old] = replace(record, expires_at=datetime.now(UTC) + grace)
        return self.keys.issue(tenant, record.name)


class FakeSpaceRegistry(FakeSpaces):
    """SpaceRegistry over FakeDb (limit 8, in-use check)."""

    max_spaces = 8

    async def register(self, spec: SpaceSpec) -> EmbeddingSpace:
        for space in self.db.spaces.values():
            if space.name == spec.name:
                if spec.calibrated_threshold is None:
                    return space
                updated = replace(space, calibrated_threshold=spec.calibrated_threshold)
                self.db.spaces[space.id] = updated
                return updated
        if len(self.db.spaces) >= self.max_spaces:
            raise ConflictError("too many spaces", code="space_limit_reached")
        space = EmbeddingSpace(
            max(self.db.spaces, default=0) + 1,
            spec.name,
            spec.provider,
            spec.model,
            spec.dimensions,
            spec.send_dimensions,
            spec.normalise,
            spec.calibrated_threshold,
        )
        return self.db.add_space(space)

    async def delete(self, name: str) -> None:
        space = await self.get(name)
        for col in self.db.cols.values():
            if space.id in (col.active_space_id, col.target_space_id):
                raise ConflictError("in use", code="space_in_use")
        self.db.spaces.pop(space.id)


class FakeExporter:
    """TenantExporter yielding the tenant and its collections."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.closed = 0

    async def export(
        self, tenant: TenantId, *, include_vectors: bool, include_originals: bool
    ) -> AsyncIterator[ExportRecord]:
        try:
            t = self.db.tenants.get(tenant)
            if t is None:
                raise NotFoundError("tenant not found")
            yield ExportRecord("tenant", {"id": str(t.id), "name": t.name})
            for (owner, cid), col in self.db.cols.items():
                if owner == tenant:
                    yield ExportRecord("collection", {"id": str(cid), "name": col.name})
            if include_vectors:
                yield ExportRecord("vector", {"chunk_id": "x", "embedding": [1.0]})
            if include_originals:
                yield ExportRecord("version", {"original_base64": "aGk="})
        finally:
            self.closed += 1


class FakeResolver:
    """HostResolver with a fixed table; unknown hosts do not resolve."""

    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self.table = table if table is not None else {"hooks.example.com": ["93.184.216.34"]}

    async def resolve(self, host: str, port: int) -> Sequence[str]:
        return self.table.get(host, [])


class FakeInspector:
    """UploadInspector: `%PDF-` → pdf (`ENCRYPTED` inside → pdf_encrypted), NUL → 415,
    `.md` → markdown, else text."""

    async def inspect(self, data: bytes, filename: str) -> InspectedFile:
        if data.startswith(b"%PDF-"):
            if b"ENCRYPTED" in data:
                raise DomainError("PDF is encrypted", code="pdf_encrypted")
            return InspectedFile("application/pdf", pages=1)
        if b"\x00" in data:
            raise DomainError("binary content", code="unsupported_media_type")
        if filename.endswith(".md"):
            return InspectedFile("text/markdown")
        return InspectedFile("text/plain")


class FakeChunkReader:
    """ChunkReader over FakeDb (current-version memberships only)."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db

    async def chunk(
        self, tenant: TenantId, collection_id: CollectionId, chunk_id: ChunkId
    ) -> ChunkDetail:
        chunk = self.db.chunks.get((tenant, chunk_id))
        col = self.db.cols.get((tenant, collection_id))
        if chunk is None or col is None or col.status != "active":
            raise NotFoundError("chunk not found", code="chunk_not_found")
        doc = self.db.docs.get((tenant, chunk.document_id))
        if doc is None or doc.current_version_id is None or chunk.collection_id != collection_id:
            raise NotFoundError("chunk not found", code="chunk_not_found")
        version = self.db.versions[(tenant, doc.current_version_id)]
        members = [
            m
            for m in self.db.members.get((tenant, doc.current_version_id), [])
            if m.chunk_id == chunk_id
        ]
        if not members:
            raise NotFoundError("chunk not found", code="chunk_not_found")
        return ChunkDetail(
            chunk.id,
            chunk.collection_id,
            chunk.document_id,
            doc.title,
            chunk.embedded_text,
            (),
            len(chunk.embedded_text.split()),
            tuple(
                ChunkPlacement(version.version, m.ordinal, m.page_from, m.page_to) for m in members
            ),
        )


class FakeDeliveries:
    """WebhookDeliveryLog listing preset records per (tenant, collection)."""

    def __init__(self) -> None:
        self.records: dict[tuple[TenantId, CollectionId], list[DeliveryRecord]] = {}

    async def attempts(self, tenant: TenantId, delivery_id: UUID) -> int:
        return 0

    async def record_attempt(self, tenant: TenantId, attempt: object) -> None:
        return None

    async def list_for_collection(
        self, tenant: TenantId, collection_id: CollectionId, limit: int = 50
    ) -> list[DeliveryRecord]:
        return self.records.get((tenant, collection_id), [])[:limit]


class TenantSearchStore(FakeSearchStore):
    """FakeSearchStore that, like RLS, hides collections the tenant does not own."""

    def __init__(self, db: FakeDb, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.db = db

    @asynccontextmanager
    async def snapshot(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> AsyncIterator[SearchSnapshot]:
        col = self.db.cols.get((tenant, collection_id))
        if col is None or col.status != "active":
            raise NotFoundError("collection not found")
        async with super().snapshot(tenant, collection_id) as snap:
            yield snap
