# SPDX-License-Identifier: Apache-2.0
"""Chunk route."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from footnote.api.auth import Tenant
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.schemas.common import error_responses
from footnote.api.schemas.documents import ChunkOut
from footnote.domain.documents import ChunkId, CollectionId

router = APIRouter(prefix="/v1/collections/{collection_id}/chunks", tags=["chunks"])


@router.get("/{chunk_id}", response_model=ChunkOut, responses=error_responses(401, 404))
async def get_chunk(
    collection_id: UUID,
    chunk_id: UUID,
    principal: Tenant,
    deps: Annotated[ApiDeps, Depends(get_deps)],
) -> ChunkOut:
    """Text, heading path, document and current-version memberships (`chunk_not_found`)."""
    detail = await deps.documents.chunk(
        principal.tenant, CollectionId(collection_id), ChunkId(chunk_id)
    )
    return ChunkOut.of(detail)
