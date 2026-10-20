# SPDX-License-Identifier: Apache-2.0
"""Corpus indexing through the upload + ingest use cases (evaluation harness)."""

from __future__ import annotations

import hashlib

import pytest

from footnote.app.documents import UploadRequest
from footnote.app.evaluation.corpus import (
    INGEST_JOB_DELAY,
    CorpusIndexError,
    index_corpus,
    upload_request,
)
from footnote.app.evaluation.datasets import CorpusDoc
from footnote.domain.documents import CollectionId, NewUpload, UploadResult
from footnote.domain.tenancy import TenantId

from .rig import Rig, build


class RigUploader:
    """`DocumentService.upload` reduced to the repository call (no inspector)."""

    def __init__(self, rig: Rig, fail_on: str | None = None) -> None:
        self.rig = rig
        self.fail_on = fail_on

    async def upload(
        self, tenant: TenantId, collection_id: CollectionId, request: UploadRequest
    ) -> UploadResult:
        if request.external_id == self.fail_on:
            raise ValueError("boom")
        return await self.rig.documents.upload(
            tenant,
            NewUpload(
                collection_id,
                request.external_id,
                request.title or "",
                {},
                "text/plain",
                hashlib.sha256(request.data).digest(),
                request.data,
                self.rig.collection.settings.chunking.settings_hash(),
                ingest_delay=request.ingest_delay,
            ),
        )


DOCS = [
    CorpusDoc("d1", "First title", "alpha " * 20),
    CorpusDoc("d2", "Second", "beta " * 20),
    CorpusDoc("d3", "Empty text", ""),
]


def test_upload_request_maps_beir_fields() -> None:
    request = upload_request(CorpusDoc("42", "T" * 600, "body"))
    assert (request.filename, request.external_id, request.data) == ("42.txt", "42", b"body")
    assert request.title == "T" * 512
    assert request.ingest_delay == INGEST_JOB_DELAY
    assert upload_request(CorpusDoc("7", "Only title", " ")).data == b"Only title"
    assert upload_request(CorpusDoc("8", "", "")).data == b"8"


async def test_index_corpus_indexes_every_document_and_maps_ids() -> None:
    rig = build(webhook=False)
    seen: list[tuple[int, int]] = []
    result = await index_corpus(
        RigUploader(rig),
        rig.ingest,
        rig.tenant,
        rig.collection.id,
        DOCS,
        concurrency=2,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert sorted(result.doc_ids.values()) == ["d1", "d2", "d3"]
    assert seen[-1] == (3, 3)
    assert result.chunks == len(rig.db.current_chunks(rig.tenant, rig.collection.id))
    assert result.embedded == result.chunks > 0
    ingest_jobs = rig.db.jobs_of(rig.tenant, "ingest")
    assert len(ingest_jobs) == 3
    assert {j.delay for j in ingest_jobs} == {INGEST_JOB_DELAY}
    # every version is current: the deferred job would be a no-op
    titles = {
        rig.db.chunks[(rig.tenant, c)].embedded_text.split("\n", 1)[0]
        for c in rig.db.current_chunks(rig.tenant, rig.collection.id)
    }
    assert titles == {"First title", "Second", "Empty text"}


async def test_index_corpus_reports_failed_documents() -> None:
    rig = build(webhook=False)
    with pytest.raises(CorpusIndexError, match=r"1 of 3 documents failed to index: d2: ValueError"):
        await index_corpus(
            RigUploader(rig, fail_on="d2"), rig.ingest, rig.tenant, rig.collection.id, DOCS
        )
