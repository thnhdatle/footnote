# SPDX-License-Identifier: Apache-2.0
"""`/v1/collections`: collections and their webhook deliveries."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response

from footnote.api.auth import Tenant
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.pagination import DEFAULT_LIMIT, Cursor, Limit, decode_cursor, next_cursor
from footnote.api.schemas.collections import (
    CollectionChangeOut,
    CollectionCreate,
    CollectionOut,
    CollectionPatch,
    DeliveryOut,
)
from footnote.api.schemas.common import JobAccepted, Page, error_responses
from footnote.domain.documents import CollectionId

router = APIRouter(prefix="/v1/collections", tags=["collections"])
Deps = Annotated[ApiDeps, Depends(get_deps)]


@router.post(
    "",
    status_code=201,
    response_model=CollectionChangeOut,
    responses=error_responses(401, 409, 422, 429),
)
async def create_collection(
    body: CollectionCreate, principal: Tenant, deps: Deps
) -> CollectionChangeOut:
    """Create a collection (settings default from config); a webhook secret is returned once."""
    change = await deps.collections.create(
        principal.tenant, body.name, body.settings_patch(), body.webhook_patch()
    )
    return CollectionChangeOut.of_change(change)


@router.get("", response_model=Page[CollectionOut], responses=error_responses(401, 422))
async def list_collections(
    principal: Tenant, deps: Deps, limit: Limit = DEFAULT_LIMIT, cursor: Cursor = None
) -> Page[CollectionOut]:
    """Collections of the tenant ordered by id."""
    after = decode_cursor(cursor)
    infos = await deps.collections.list_page(
        principal.tenant, None if after is None else CollectionId(after), limit + 1
    )
    return Page[CollectionOut](
        data=[CollectionOut.of(i) for i in infos[:limit]],
        next_cursor=next_cursor([i.record.id for i in infos], limit),
    )


@router.get("/{collection_id}", response_model=CollectionOut, responses=error_responses(401, 404))
async def get_collection(collection_id: UUID, principal: Tenant, deps: Deps) -> CollectionOut:
    """A collection with counts, active space, re-embed progress and jobs in progress."""
    info = await deps.collections.get(principal.tenant, CollectionId(collection_id))
    return CollectionOut.of(info)


@router.patch(
    "/{collection_id}",
    response_model=CollectionChangeOut,
    responses={202: {"model": CollectionChangeOut}, **error_responses(401, 404, 422)},
)
async def update_collection(
    collection_id: UUID,
    body: CollectionPatch,
    principal: Tenant,
    deps: Deps,
    response: Response,
) -> CollectionChangeOut:
    """Change settings: 202 with `job_id` when a reindex, fts_reconfig or reembed job starts."""
    change = await deps.collections.update(
        principal.tenant, CollectionId(collection_id), body.settings_patch(), body.webhook_patch()
    )
    if change.job_ids:
        response.status_code = 202
    return CollectionChangeOut.of_change(change)


@router.delete(
    "/{collection_id}",
    status_code=202,
    response_model=JobAccepted,
    responses=error_responses(401, 404),
)
async def delete_collection(collection_id: UUID, principal: Tenant, deps: Deps) -> JobAccepted:
    """Hide the collection now; a `collection_delete` job removes its rows."""
    job_id = await deps.collections.delete(principal.tenant, CollectionId(collection_id))
    return JobAccepted(job_id=job_id)


@router.get(
    "/{collection_id}/webhook-deliveries",
    response_model=Page[DeliveryOut],
    responses=error_responses(401, 404),
)
async def list_deliveries(
    collection_id: UUID,
    principal: Tenant,
    deps: Deps,
    limit: Annotated[int, Query(ge=1, le=500)] = DEFAULT_LIMIT,
) -> Page[DeliveryOut]:
    """Newest webhook delivery attempts first (one page; `next_cursor` is always null)."""
    records = await deps.collections.deliveries(
        principal.tenant, CollectionId(collection_id), limit
    )
    return Page[DeliveryOut](data=[DeliveryOut.of(d) for d in records])
