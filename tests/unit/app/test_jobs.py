# SPDX-License-Identifier: Apache-2.0
"""GC, reindex, webhook jobs and the per-kind job handlers over the fakes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

import pytest

from footnote.app.gc import GcOutcome
from footnote.app.webhooks import WebhookNotifier
from footnote.domain.documents import CollectionId, DocumentId, JobId, VersionStatus
from footnote.domain.errors import DomainError, UpstreamError, ValidationError
from footnote.domain.jobs import JOB_KINDS, NewJob, PermanentJobError
from footnote.domain.settings import ChunkingSettings, Language
from footnote.domain.tenancy import TenantStatus
from footnote.domain.webhooks import WebhookEvent, WebhookSettings

from .fakes import FakeClock, FakeJobRun
from .rig import CHUNKING, SPACE_A, SPACE_B, Rig, build, paragraph

P = [paragraph(f"g{n}y") for n in range(6)]


async def _run(rig: Rig, kind: str, args: dict[str, Any] | None = None) -> FakeJobRun:
    """Enqueue (when args are given) and run the oldest queued job of `kind`."""
    if args is not None:
        await rig.jobs.enqueue(rig.tenant, NewJob(kind, args))
    run = rig.jobs.claim(rig.tenant, kind)
    assert run is not None
    await rig.handlers.handlers()[kind](run)
    return run


async def test_gc_removes_superseded_chunks_and_keeps_shared_ones() -> None:
    rig = build(retain_originals=False)
    v1 = await rig.index_texts(P[:3])
    v2 = await rig.index_texts([P[0], P[3]])
    assert len(rig.db.chunks) == 4
    assert (rig.tenant, v1.version_id) not in rig.db.blobs  # dropped when v1 became ready
    rig.db.blobs[(rig.tenant, v1.version_id)] = b"left by a crash"  # type: ignore[index]

    outcome = await rig.gc.run(rig.tenant, {"document_id": str(v2.document_id)})

    assert outcome == GcOutcome(memberships=3, chunks=2, blobs=1)
    assert rig.current_texts() == {P[0], P[3]}
    assert len(rig.db.chunks) == 2
    assert {k[1] for k in rig.db.vectors} == rig.db.current_chunks(rig.tenant, rig.collection.id)
    assert (rig.tenant, v1.version_id) not in rig.db.blobs
    rig.db.search_view(rig.tenant, rig.collection.id)


async def test_gc_keeps_originals_when_retained_and_spares_inflight_chunks() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    await rig.index_texts(P[2:4])
    inflight = await rig.upload(P[4:5])
    rig.db.versions[(rig.tenant, inflight.version_id)].elements = []  # type: ignore[index]

    outcome = await rig.gc.run(rig.tenant, {"document_id": str(inflight.document_id)})

    assert (outcome.memberships, outcome.chunks, outcome.blobs) == (2, 0, 0)
    assert len(rig.db.chunks) == 4


async def test_gc_space_removes_only_unused_space_vectors() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    target = await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_B.name)
    assert target is not None
    await rig.reembed.run(rig.tenant, rig.collection.id, SPACE_B.id)
    args = {"collection_id": str(rig.collection.id)}

    assert await rig.gc.run(rig.tenant, {**args, "space_id": SPACE_B.id}) == GcOutcome()
    assert await rig.gc.run(rig.tenant, {**args, "space_id": SPACE_A.id}) == GcOutcome(vectors=2)
    assert {k[2] for k in rig.db.vectors} == {SPACE_B.id}


async def test_gc_of_missing_rows_is_a_noop_and_bad_args_are_rejected() -> None:
    rig = build()
    missing = {"document_id": str(UUID(int=3))}
    assert await rig.gc.run(rig.tenant, missing) == GcOutcome()
    with pytest.raises(ValidationError):
        await rig.gc.run(rig.tenant, {"space_id": 1})


async def test_reindex_rechunks_changed_documents_only_once() -> None:
    rig = build()
    await rig.index_texts(P[:2], external_id="a")
    await rig.index_texts(P[2:4], external_id="b")
    await rig.index_texts(P[4:5], external_id="c")
    new_chunking = replace(CHUNKING, max_tokens=64)
    settings = replace(rig.collection.settings, chunking=new_chunking)
    await rig.collections.update(rig.tenant, rig.collection.id, settings)
    rig.embedders.calls.clear()
    for job in rig.db.jobs_of(rig.tenant, "ingest"):
        job.status = "done"  # index_texts ran them directly
    run = await _run(rig, "reindex", {"collection_id": str(rig.collection.id)})

    assert run.progress_calls == [(0, 3), (1, 3), (2, 3), (3, 3)]
    ingest_jobs = [j for j in rig.db.jobs_of(rig.tenant, "ingest") if j.status == "queued"]
    assert len(ingest_jobs) == 3
    for _ in ingest_jobs:
        await _run(rig, "ingest")
    # Two paragraphs now fit one chunk: a and b get one new chunk each, c is reused.
    assert len(rig.embedders.texts()) == 2
    assert rig.parser.calls == 3  # only the original uploads were parsed
    for doc in rig.db.docs.values():
        version = rig.db.versions[(rig.tenant, doc.current_version_id)]  # type: ignore[index]
        assert version.chunking_hash == new_chunking.settings_hash()
        assert (rig.tenant, version.id) in rig.db.blobs

    again = await rig.reindex.reindex(rig.tenant, rig.collection.id)
    assert (again.documents, again.versions) == (3, 0)


async def test_reindex_skips_documents_with_an_upload_in_flight() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    await rig.upload(P[2:3])
    settings = replace(rig.collection.settings, chunking=ChunkingSettings())
    await rig.collections.update(rig.tenant, rig.collection.id, settings)
    outcome = await rig.reindex.reindex(rig.tenant, rig.collection.id)
    assert (outcome.documents, outcome.versions) == (1, 0)


async def test_reindex_of_missing_collection_or_deleted_document() -> None:
    rig = build()
    outcome = await rig.reindex.reindex(rig.tenant, CollectionId(UUID(int=4)))
    assert (outcome.documents, outcome.versions) == (0, 0)

    up = await rig.index_texts(P[:1])
    settings = replace(rig.collection.settings, chunking=ChunkingSettings())
    await rig.collections.update(rig.tenant, rig.collection.id, settings)

    async def vanish(tenant: Any, document_id: Any, chunking_hash: str) -> None:
        await rig.documents.delete(rig.tenant, rig.collection.id, up.document_id)
        await original(tenant, document_id, chunking_hash)

    original = rig.documents.reversion
    rig.documents.reversion = vanish  # type: ignore[method-assign,assignment]
    outcome = await rig.reindex.reindex(rig.tenant, rig.collection.id)
    assert (outcome.documents, outcome.versions) == (1, 0)


async def test_fts_reconfig_switches_language() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    await _run(
        rig, "fts_reconfig", {"collection_id": str(rig.collection.id), "language": "english"}
    )
    assert rig.db.col(rig.tenant, rig.collection.id).settings.language is Language.ENGLISH
    assert {c.language for c in rig.db.chunks.values()} == {Language.ENGLISH}
    # A collection deleted meanwhile is a no-op.
    await rig.reindex.fts_reconfig(rig.tenant, CollectionId(UUID(int=4)), Language.SIMPLE)


async def test_handlers_cover_every_kind_but_webhook() -> None:
    rig = build()
    assert set(rig.handlers.handlers()) == JOB_KINDS - {"webhook"}


async def test_ingest_handler_runs_the_uploaded_version() -> None:
    rig = build()
    await rig.upload(P[:2])
    await _run(rig, "ingest")
    assert rig.current_texts() == set(P[:2])


async def test_ingest_handler_retries_then_fails_for_good_on_last_attempt() -> None:
    rig = build()
    up = await rig.upload(P[:2])
    rig.embedders.error = UpstreamError("down")
    job = rig.db.jobs_of(rig.tenant, "ingest")[0]
    for attempt in (1, 2):
        run = rig.jobs.claim(rig.tenant, "ingest")
        assert run is not None
        with pytest.raises(UpstreamError):
            await rig.handlers.ingest(run)
        assert job.attempts == attempt
        job.status = "queued"
    run = rig.jobs.claim(rig.tenant, "ingest")
    assert run is not None
    with pytest.raises(PermanentJobError, match="upstream_failed"):
        await rig.handlers.ingest(run)
    version = rig.db.versions[(rig.tenant, up.version_id)]  # type: ignore[index]
    assert version.status is VersionStatus.FAILED


async def test_ingest_handler_maps_permanent_failures() -> None:
    rig = build(parser_error=DomainError("bad", code="parse_failed"))
    await rig.upload(P[:1])
    with pytest.raises(PermanentJobError, match="parse_failed: bad"):
        await _run(rig, "ingest")


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("ingest", {"collection_id": "x"}),
        ("reindex", {}),
        ("fts_reconfig", {"collection_id": str(UUID(int=1)), "language": "klingon"}),
        ("fts_reconfig", {"collection_id": str(UUID(int=1))}),
        ("reembed", {"collection_id": str(UUID(int=1))}),
        ("reembed", {"collection_id": str(UUID(int=1)), "target_space_id": "two"}),
        ("gc", {}),
        ("gc", {"collection_id": str(UUID(int=1)), "space_id": "x"}),
        ("collection_delete", {"collection_id": None}),
    ],
)
async def test_invalid_args_fail_permanently(kind: str, args: dict[str, Any]) -> None:
    rig = build()
    with pytest.raises(PermanentJobError):
        await _run(rig, kind, args)


async def test_reembed_handler_reports_progress() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_B.name)
    run = await _run(rig, "reembed")
    assert run.progress_calls[-1] == (3, 3)
    assert run.row.progress == {"done": 3, "total": 3}
    assert rig.db.col(rig.tenant, rig.collection.id).active_space_id == SPACE_B.id


async def test_gc_handler_runs_document_gc() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    await rig.index_texts(P[2:3])
    await _run(rig, "gc")
    assert len(rig.db.chunks) == 1


async def test_collection_delete_requires_deleting_status() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    args = {"collection_id": str(rig.collection.id)}
    with pytest.raises(PermanentJobError, match="not being deleted"):
        await _run(rig, "collection_delete", args)
    await rig.collections.mark_deleting(rig.tenant, rig.collection.id)
    await _run(rig, "collection_delete", args)
    assert rig.db.cols == {}
    assert rig.db.chunks == {}
    # Already gone: done.
    await _run(rig, "collection_delete", args)


async def test_tenant_delete_requires_deleting_status() -> None:
    rig = build()
    await rig.index_texts(P[:2])
    with pytest.raises(PermanentJobError, match="not being deleted"):
        await _run(rig, "tenant_delete", {})
    assert rig.tenants.deleted == []
    tenant = rig.db.tenants[rig.tenant]
    rig.db.tenants[rig.tenant] = replace(tenant, status=TenantStatus.DELETING)
    await _run(rig, "tenant_delete", {})
    assert rig.tenants.deleted == [rig.tenant]
    assert rig.db.docs == {}
    assert rig.db.jobs == {}


async def test_tenant_delete_of_a_missing_tenant_is_done() -> None:
    rig = build()
    await rig.jobs.enqueue(rig.tenant, NewJob("tenant_delete"))
    run = rig.jobs.claim(rig.tenant, "tenant_delete")
    assert run is not None
    rig.db.tenants.clear()
    await rig.handlers.tenant_delete(run)
    assert rig.tenants.deleted == []


async def test_webhook_notifier_builds_signed_payload_fields() -> None:
    rig = build()
    notifier = WebhookNotifier(rig.jobs, FakeClock())
    up = await rig.upload(P[:1])
    version = await rig.documents.version(rig.tenant, up.version_id)  # type: ignore[arg-type]
    job = notifier.job_for(rig.collection, WebhookEvent.DOCUMENT_FAILED, version, {"code": "x"})
    assert job is not None
    assert job.kind == "webhook"
    assert dict(job.args) == {
        "event": "document.failed",
        "collection_id": str(rig.collection.id),
        "document_id": str(up.document_id),
        "version": 1,
        "status": "failed",
        "error": {"code": "x"},
        "occurred_at": "2023-11-14T22:13:20+00:00",
    }
    job_id = await notifier.notify(rig.tenant, rig.collection, WebhookEvent.DOCUMENT_READY, version)
    assert isinstance(job_id, type(JobId(UUID(int=0))))
    unsubscribed = replace(
        rig.collection,
        settings=replace(rig.collection.settings, webhook=WebhookSettings(())),
    )
    assert (
        await notifier.notify(rig.tenant, unsubscribed, WebhookEvent.DOCUMENT_READY, version)
        is None
    )
    no_secret = replace(rig.collection, webhook_secret=None)
    assert notifier.job_for(no_secret, WebhookEvent.DOCUMENT_READY, version) is None


def test_document_id_type() -> None:
    assert DocumentId(UUID(int=1)) == UUID(int=1)
