# SPDX-License-Identifier: Apache-2.0
"""Index a BEIR corpus through Footnote's pipeline.

Each corpus document becomes one plain-text upload: the BEIR title is the document title
(it leads every chunk's embedded text), the BEIR text is the file, and the BEIR id is the
`external_id`, so chunks pool back to BEIR documents. Uploads go through the
`DocumentService` (checks, blob, version) and each version is indexed in-process by the
`IngestService` (parse in the sandbox → chunk → embed → swap) with bounded concurrency.
The queued `ingest` job is deferred (`INGEST_JOB_DELAY`) so no worker races the harness;
it is an idempotent no-op if it ever runs, and goes with the temporary tenant.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from footnote.app.documents import UploadRequest
from footnote.app.evaluation.datasets import CorpusDoc
from footnote.app.ingest import IngestFailedError, IngestService, IngestTarget, is_transient
from footnote.domain.documents import CollectionId, DocumentId, UploadResult
from footnote.domain.tenancy import TenantId

INGEST_JOB_DELAY = timedelta(days=7)
TITLE_MAX = 512
INGEST_ATTEMPTS = 3

type Progress = Callable[[int, int], None]


class Uploader(Protocol):
    """`DocumentService.upload`."""

    async def upload(
        self, tenant: TenantId, collection_id: CollectionId, request: UploadRequest
    ) -> UploadResult:
        """Store an upload and create its version."""


class CorpusIndexError(RuntimeError):
    """Documents failed to index; the evaluation would be incomplete."""


@dataclass(frozen=True)
class IndexedCorpus:
    """Footnote document id → BEIR document id, and the ingest totals."""

    doc_ids: dict[DocumentId, str]
    chunks: int
    embedded: int
    seconds: float


def upload_request(doc: CorpusDoc) -> UploadRequest:
    """The upload of one corpus document (plain text; title and id from BEIR)."""
    body = doc.text if doc.text.strip() else doc.title
    return UploadRequest(
        filename=f"{doc.doc_id}.txt",
        data=(body or doc.doc_id).encode("utf-8"),
        external_id=doc.doc_id,
        title=doc.title[:TITLE_MAX],
        ingest_delay=INGEST_JOB_DELAY,
    )


async def _ingest(ingest: IngestService, tenant: TenantId, target: IngestTarget) -> tuple[int, int]:
    for attempt in range(1, INGEST_ATTEMPTS + 1):
        try:
            outcome = await ingest.ingest(tenant, target, last_attempt=attempt == INGEST_ATTEMPTS)
        except IngestFailedError:
            raise
        except Exception as exc:
            if attempt == INGEST_ATTEMPTS or not is_transient(exc):
                raise
            await asyncio.sleep(0.5 * attempt)
        else:
            return outcome.chunks, outcome.embedded
    raise AssertionError("unreachable")  # pragma: no cover


async def index_corpus(  # noqa: PLR0913 - the indexing inputs
    uploader: Uploader,
    ingest: IngestService,
    tenant: TenantId,
    collection_id: CollectionId,
    corpus: Sequence[CorpusDoc],
    *,
    concurrency: int = 8,
    progress: Progress | None = None,
    clock: Callable[[], float] | None = None,
) -> IndexedCorpus:
    """Upload and index every document; CorpusIndexError lists the failures."""
    loop = asyncio.get_running_loop()
    now = clock or loop.time
    start = now()
    gate = asyncio.Semaphore(max(1, concurrency))
    doc_ids: dict[DocumentId, str] = {}
    failures: list[str] = []
    totals = [0, 0]
    done = 0

    async def one(doc: CorpusDoc) -> None:
        nonlocal done
        async with gate:
            try:
                result = await uploader.upload(tenant, collection_id, upload_request(doc))
                doc_ids[result.document_id] = doc.doc_id
                if result.version_id is not None:
                    target = IngestTarget(collection_id, result.document_id, result.version_id)
                    chunks, embedded = await _ingest(ingest, tenant, target)
                    totals[0] += chunks
                    totals[1] += embedded
            except Exception as exc:  # collected: one bad document must not hide the others
                failures.append(f"{doc.doc_id}: {type(exc).__name__}: {exc}")
            done += 1
            if progress is not None:
                progress(done, len(corpus))

    await asyncio.gather(*(one(doc) for doc in corpus))
    if failures:
        shown = "; ".join(failures[:5])
        msg = f"{len(failures)} of {len(corpus)} documents failed to index: {shown}"
        raise CorpusIndexError(msg)
    return IndexedCorpus(doc_ids, totals[0], totals[1], now() - start)
