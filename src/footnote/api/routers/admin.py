# SPDX-License-Identifier: Apache-2.0
"""Admin routes for tenants, keys and embedding spaces."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Annotated
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse

from footnote.api.auth import Admin
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.pagination import (
    DEFAULT_LIMIT,
    Cursor,
    Limit,
    decode_cursor,
    decode_int_cursor,
    next_cursor,
)
from footnote.api.schemas.admin import (
    IssuedKeyOut,
    KeyCreate,
    KeyOut,
    QuotasIn,
    SpaceIn,
    SpaceOut,
    TenantCreate,
    TenantOut,
    TenantPatch,
    UsageOut,
)
from footnote.api.schemas.common import JobAccepted, Page, error_responses
from footnote.domain.tenancy import ExportRecord, Quotas, TenantId

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Admin])
Deps = Annotated[ApiDeps, Depends(get_deps)]
NDJSON = "application/x-ndjson"


async def _close(records: AsyncIterator[ExportRecord]) -> None:
    close = getattr(records, "aclose", None)
    if close is not None:
        await close()


def _quotas(base: Quotas, patch: QuotasIn | None) -> Quotas | None:
    if patch is None:
        return None
    return replace(base, **patch.changes())


@router.post(
    "/tenants", status_code=201, response_model=TenantOut, responses=error_responses(401, 422)
)
async def create_tenant(body: TenantCreate, deps: Deps) -> TenantOut:
    """Create a tenant (quotas default to `tenancy.default_quotas`)."""
    defaults = deps.admin.default_quotas
    tenant = await deps.admin.create_tenant(body.name, _quotas(defaults, body.quotas))
    return TenantOut.of(tenant)


@router.get("/tenants", response_model=Page[TenantOut], responses=error_responses(401, 422))
async def list_tenants(
    deps: Deps, limit: Limit = DEFAULT_LIMIT, cursor: Cursor = None
) -> Page[TenantOut]:
    """Tenants ordered by id."""
    after = decode_cursor(cursor)
    tenants = await deps.admin.list_tenants(None if after is None else TenantId(after), limit + 1)
    return Page[TenantOut](
        data=[TenantOut.of(t) for t in tenants[:limit]],
        next_cursor=next_cursor([t.id for t in tenants], limit),
    )


@router.get("/tenants/{tenant_id}", response_model=TenantOut, responses=error_responses(401, 404))
async def get_tenant(tenant_id: UUID, deps: Deps) -> TenantOut:
    """One tenant."""
    return TenantOut.of(await deps.admin.get_tenant(TenantId(tenant_id)))


@router.patch(
    "/tenants/{tenant_id}", response_model=TenantOut, responses=error_responses(401, 404, 422)
)
async def update_tenant(tenant_id: UUID, body: TenantPatch, deps: Deps) -> TenantOut:
    """Change the name and/or quotas (omitted quota fields keep their value)."""
    tenant = TenantId(tenant_id)
    quotas = None
    if body.quotas is not None:
        current = await deps.admin.get_tenant(tenant)
        quotas = _quotas(current.quotas, body.quotas)
    return TenantOut.of(await deps.admin.update_tenant(tenant, name=body.name, quotas=quotas))


@router.delete(
    "/tenants/{tenant_id}",
    status_code=202,
    response_model=JobAccepted,
    responses=error_responses(401, 404),
)
async def delete_tenant(tenant_id: UUID, deps: Deps) -> JobAccepted:
    """Revoke every key now; data, keys and the tenant row go with a `tenant_delete` job."""
    return JobAccepted(job_id=await deps.admin.delete_tenant(TenantId(tenant_id)))


@router.get(
    "/tenants/{tenant_id}/usage", response_model=UsageOut, responses=error_responses(401, 404)
)
async def tenant_usage(tenant_id: UUID, deps: Deps) -> UsageOut:
    """Collections, documents, chunks, vectors and stored bytes of a tenant."""
    return UsageOut.of(await deps.admin.usage(TenantId(tenant_id)))


@router.get(
    "/tenants/{tenant_id}/export",
    response_model=None,
    responses={
        200: {
            "content": {NDJSON: {"schema": {"type": "string"}}},
            "description": 'One `{"type", "data"}` JSON object per line',
        },
        **error_responses(401, 404),
    },
)
async def export_tenant(
    tenant_id: UUID,
    deps: Deps,
    include_vectors: Annotated[bool, Query()] = False,
    include_originals: Annotated[bool, Query()] = False,
) -> Response:
    """Stream the tenant's collections (no webhook secrets), documents, versions,
    elements, chunks and memberships; vectors and originals (base64) on request."""
    records = await deps.admin.export(
        TenantId(tenant_id),
        include_vectors=include_vectors,
        include_originals=include_originals,
    )
    try:
        first = await anext(records, None)
    except BaseException:
        await _close(records)
        raise

    async def lines() -> AsyncIterator[bytes]:
        try:
            record: ExportRecord | None = first
            while record is not None:
                yield orjson.dumps({"type": record.type, "data": record.data}) + b"\n"
                record = await anext(records, None)
        finally:
            await _close(records)

    return StreamingResponse(lines(), media_type=NDJSON)


@router.post(
    "/tenants/{tenant_id}/keys",
    status_code=201,
    response_model=IssuedKeyOut,
    responses=error_responses(401, 404, 422),
)
async def create_key(tenant_id: UUID, body: KeyCreate, deps: Deps) -> IssuedKeyOut:
    """Issue an API key; the key is returned once."""
    issued = await deps.admin.create_key(TenantId(tenant_id), body.name, body.expires_at)
    return IssuedKeyOut.of_issued(issued)


@router.get(
    "/tenants/{tenant_id}/keys",
    response_model=Page[KeyOut],
    responses=error_responses(401, 404, 422),
)
async def list_keys(
    tenant_id: UUID, deps: Deps, limit: Limit = DEFAULT_LIMIT, cursor: Cursor = None
) -> Page[KeyOut]:
    """Keys of a tenant ordered by id: prefix, name, timestamps, revoked (never the key)."""
    keys = await deps.admin.list_keys(TenantId(tenant_id), decode_cursor(cursor), limit + 1)
    return Page[KeyOut](
        data=[KeyOut.of(k) for k in keys[:limit]],
        next_cursor=next_cursor([k.id for k in keys], limit),
    )


@router.delete(
    "/tenants/{tenant_id}/keys/{key_id}", status_code=204, responses=error_responses(401, 404)
)
async def revoke_key(tenant_id: UUID, key_id: UUID, deps: Deps) -> Response:
    """Revoke a key now."""
    await deps.admin.revoke_key(TenantId(tenant_id), key_id)
    return Response(status_code=204)


@router.post(
    "/tenants/{tenant_id}/keys/{key_id}/rotate",
    status_code=201,
    response_model=IssuedKeyOut,
    responses=error_responses(401, 404),
)
async def rotate_key(tenant_id: UUID, key_id: UUID, deps: Deps) -> IssuedKeyOut:
    """Issue a replacement (returned once); the old key stays valid for
    `auth.key_rotation_grace`."""
    return IssuedKeyOut.of_issued(await deps.admin.rotate_key(TenantId(tenant_id), key_id))


@router.post(
    "/embedding-spaces",
    status_code=201,
    response_model=SpaceOut,
    responses=error_responses(401, 409, 422),
)
async def register_space(body: SpaceIn, deps: Deps) -> SpaceOut:
    """Register a space and its partial HNSW index (idempotent by name)."""
    return SpaceOut.of(await deps.admin.register_space(body.spec()))


@router.get("/embedding-spaces", response_model=Page[SpaceOut], responses=error_responses(401, 422))
async def list_spaces(
    deps: Deps, limit: Limit = DEFAULT_LIMIT, cursor: Cursor = None
) -> Page[SpaceOut]:
    """Registered spaces ordered by id."""
    spaces = await deps.admin.list_spaces(decode_int_cursor(cursor), limit + 1)
    return Page[SpaceOut](
        data=[SpaceOut.of(s) for s in spaces[:limit]],
        next_cursor=next_cursor([s.id for s in spaces], limit),
    )


@router.delete(
    "/embedding-spaces/{name}", status_code=204, responses=error_responses(401, 404, 409)
)
async def delete_space(name: str, deps: Deps) -> Response:
    """Drop a space and its index (409 `space_in_use` while a collection uses it)."""
    await deps.admin.delete_space(name)
    return Response(status_code=204)
