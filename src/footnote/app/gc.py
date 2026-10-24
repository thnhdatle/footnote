# SPDX-License-Identifier: Apache-2.0
"""Garbage collection after swaps and switches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from footnote.domain.documents import CollectionId, DocumentId, GcStats
from footnote.domain.errors import NotFoundError, ValidationError
from footnote.domain.ports import IndexRepository
from footnote.domain.tenancy import TenantId


@dataclass(frozen=True, slots=True)
class GcOutcome:
    memberships: int = 0
    chunks: int = 0
    blobs: int = 0
    vectors: int = 0


class GcService:
    """Runs `gc` jobs."""

    def __init__(self, index: IndexRepository, *, retain_originals: bool = True) -> None:
        self._index = index
        self._retain = retain_originals

    async def run(self, tenant: TenantId, args: Mapping[str, Any]) -> GcOutcome:
        """Removes the rows that the job arguments name."""
        try:
            if "document_id" in args:
                stats: GcStats = await self._index.gc_document(
                    tenant,
                    DocumentId(UUID(str(args["document_id"]))),
                    delete_superseded_originals=not self._retain,
                )
                return GcOutcome(stats.memberships, stats.chunks, stats.blobs)
            if "collection_id" in args and "space_id" in args:
                vectors = await self._index.gc_space(
                    tenant,
                    CollectionId(UUID(str(args["collection_id"]))),
                    int(args["space_id"]),
                )
                return GcOutcome(vectors=vectors)
        except NotFoundError:
            return GcOutcome()
        raise ValidationError("args", "gc needs document_id or collection_id and space_id")
