# SPDX-License-Identifier: Apache-2.0
"""Re-chunks documents after a chunking change."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from footnote.domain.documents import CollectionId, DocumentId
from footnote.domain.errors import NotFoundError
from footnote.domain.ports import CollectionRepository, DocumentRepository
from footnote.domain.settings import Language
from footnote.domain.tenancy import TenantId

type Progress = Callable[[int, int], Awaitable[None]]

PAGE_SIZE = 500


@dataclass(frozen=True, slots=True)
class ReindexOutcome:
    documents: int
    versions: int


async def _no_progress(done: int, total: int) -> None: ...


class ReindexService:
    """Runs `reindex` and `fts_reconfig` jobs."""

    def __init__(
        self,
        collections: CollectionRepository,
        documents: DocumentRepository,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self._collections = collections
        self._documents = documents
        self._page_size = page_size

    async def reindex(
        self, tenant: TenantId, collection_id: CollectionId, progress: Progress = _no_progress
    ) -> ReindexOutcome:
        """Creates a new version for each document chunked with old settings."""
        try:
            collection = await self._collections.get(tenant, collection_id)
        except NotFoundError:
            return ReindexOutcome(0, 0)
        chunking_hash = collection.settings.chunking.settings_hash()
        ids = await self._document_ids(tenant, collection_id)
        created = 0
        await progress(0, len(ids))
        for done, document_id in enumerate(ids, start=1):
            try:
                result = await self._documents.reversion(tenant, document_id, chunking_hash)
            except NotFoundError:
                result = None
            created += result is not None
            await progress(done, len(ids))
        return ReindexOutcome(len(ids), created)

    async def _document_ids(
        self, tenant: TenantId, collection_id: CollectionId
    ) -> list[DocumentId]:
        ids: list[DocumentId] = []
        after: DocumentId | None = None
        while True:
            page = await self._documents.list_page(tenant, collection_id, after, self._page_size)
            ids.extend(d.id for d in page if d.current_version_id is not None)
            if len(page) < self._page_size:
                return ids
            after = page[-1].id

    async def fts_reconfig(
        self, tenant: TenantId, collection_id: CollectionId, language: Language
    ) -> None:
        """Switches the full-text language of a collection."""
        try:
            await self._collections.set_language(tenant, collection_id, language)
        except NotFoundError:
            return
