# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from footnote.app.ingest import (
    IngestFailedError,
    IngestStatus,
    error_body,
    is_transient,
    page_count,
)
from footnote.domain.chunking import content_hash
from footnote.domain.documents import (
    DocumentStatus,
    Element,
    ElementKind,
    Membership,
    VersionStatus,
)
from footnote.domain.errors import (
    ConflictError,
    DomainError,
    QuotaExceededError,
    UpstreamError,
    ValidationError,
)
from footnote.domain.tenancy import Quotas
from footnote.domain.webhooks import WebhookEvent, WebhookSettings

from .rig import GC_GRACE, SPACE_A, SPACE_B, Rig, build, paragraph

P = [paragraph(w) for w in ("alpha", "beta", "gamma", "delta", "eps", "zeta")]


def _webhook_jobs(rig: Rig) -> list[dict[str, Any]]:
    return [j.args for j in rig.db.jobs_of(rig.tenant, "webhook")]


async def test_ingest_indexes_a_new_document() -> None:
    rig = build()
    up = await rig.upload(P[:3])
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))

    assert outcome.status is IngestStatus.CURRENT
    assert (outcome.chunks, outcome.new, outcome.reused, outcome.embedded) == (3, 3, 0, 3)
    assert rig.current_texts() == set(P[:3])
    assert rig.db.search_view(rig.tenant, rig.collection.id)
    doc = await rig.documents.get(rig.tenant, rig.collection.id, up.document_id)
    assert doc.status is DocumentStatus.READY
    assert doc.current_version == 1
    version = await rig.documents.version(rig.tenant, up.version_id)  # type: ignore[arg-type]
    assert version.status is VersionStatus.READY
    assert version.pages == 3
    assert rig.db.col(rig.tenant, rig.collection.id).chunk_count == 3
    assert [len(ts) for _, ts in rig.embedders.calls] == [2, 1]
    assert {s for s, _ in rig.embedders.calls} == {SPACE_A.id}
    gc = rig.db.jobs_of(rig.tenant, "gc")
    assert [(j.args, j.delay) for j in gc] == [({"document_id": str(up.document_id)}, GC_GRACE)]
    [hook] = _webhook_jobs(rig)
    assert hook["event"] == "document.ready"
    assert hook["status"] == "ready"
    assert hook["version"] == 1
    assert hook["document_id"] == str(up.document_id)
    assert hook["collection_id"] == str(rig.collection.id)
    assert hook["error"] is None
    assert hook["occurred_at"].startswith("2023-11-14T")
    assert rig.stages == ["parse", "chunk", "embed", "swap"]


async def test_identical_reupload_is_unchanged_and_embeds_nothing() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    rig.embedders.calls.clear()

    again = await rig.upload(P[:3])

    assert again.unchanged
    assert again.job_id is None
    assert rig.embedders.calls == []
    assert len(rig.db.jobs_of(rig.tenant, "ingest")) == 1


async def test_new_version_with_same_chunks_embeds_nothing() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    rig.embedders.calls.clear()
    data_texts = [P[0], "", P[1], P[2]]
    up = await rig.upload(data_texts)
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))

    assert not up.unchanged
    assert outcome.status is IngestStatus.CURRENT
    assert (outcome.reused, outcome.new, outcome.embedded) == (3, 0, 0)
    assert rig.embedders.calls == []


async def test_changed_paragraph_embeds_only_the_new_chunk() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    rig.embedders.calls.clear()

    up = await rig.upload([P[0], P[3], P[2]])
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))

    assert (outcome.reused, outcome.new, outcome.embedded) == (2, 1, 1)
    assert [t.rsplit("\n\n", 1)[1] for t in rig.embedders.texts()] == [P[3]]
    assert rig.current_texts() == {P[0], P[3], P[2]}
    assert rig.db.col(rig.tenant, rig.collection.id).chunk_count == 3


async def test_repeated_passage_shares_one_chunk() -> None:
    rig = build()
    up = await rig.index_texts([P[0], P[1], P[0]])
    members = rig.db.members[(rig.tenant, up.version_id)]  # type: ignore[index]
    assert [m.ordinal for m in members] == [0, 1, 2]
    assert members[0].chunk_id == members[2].chunk_id
    assert len(rig.embedders.texts()) == 2
    assert rig.db.col(rig.tenant, rig.collection.id).chunk_count == 2


@settings(max_examples=40)
@given(
    versions=st.lists(
        st.lists(st.sampled_from(range(len(P))), min_size=1, max_size=5), min_size=1, max_size=5
    )
)
async def test_property_incremental_embedding_and_swap(versions: list[list[int]]) -> None:
    rig = build()
    seen: set[str] = set()
    for picks in versions:
        texts = [P[i] for i in picks]
        rig.embedders.calls.clear()
        up = await rig.upload(texts)
        if up.unchanged:
            assert rig.embedders.calls == []
            continue
        outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))
        distinct = set(texts)
        assert outcome.reused + outcome.new == len(distinct)
        assert outcome.embedded == len(distinct - seen)
        seen |= distinct
        assert rig.current_texts() == distinct
        assert set(rig.db.search_view(rig.tenant, rig.collection.id)) == rig.db.current_chunks(
            rig.tenant, rig.collection.id
        )
        assert rig.db.col(rig.tenant, rig.collection.id).chunk_count == len(distinct)


async def test_older_version_finishing_late_is_superseded() -> None:
    rig = build()
    v1 = await rig.upload(P[:2])
    v2 = await rig.upload(P[2:4])
    assert await rig.ingest.ingest(rig.tenant, rig.target(v2)) is not None
    late = await rig.ingest.ingest(rig.tenant, rig.target(v1))

    assert late.status is IngestStatus.SUPERSEDED
    assert rig.current_texts() == set(P[2:4])
    doc = await rig.documents.get(rig.tenant, rig.collection.id, v1.document_id)
    assert doc.status is DocumentStatus.READY
    assert doc.current_version == 2
    assert [h["version"] for h in _webhook_jobs(rig)] == [2]
    assert len(rig.db.jobs_of(rig.tenant, "gc")) == 2


async def test_rerun_of_a_finished_version_is_skipped() -> None:
    rig = build()
    up = await rig.index_texts(P[:2])
    rig.embedders.calls.clear()
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert outcome.status is IngestStatus.SKIPPED
    assert rig.embedders.calls == []
    assert len(rig.index.swaps) == 1


async def test_deleted_document_or_collection_is_skipped() -> None:
    rig = build()
    up = await rig.upload(P[:2])
    await rig.documents.delete(rig.tenant, rig.collection.id, up.document_id)
    assert (await rig.ingest.ingest(rig.tenant, rig.target(up))).status is IngestStatus.SKIPPED

    rig = build()
    up = await rig.upload(P[:2])
    await rig.collections.mark_deleting(rig.tenant, rig.collection.id)
    assert (await rig.ingest.ingest(rig.tenant, rig.target(up))).status is IngestStatus.SKIPPED


async def test_document_deleted_during_ingest_is_skipped() -> None:
    rig = build()
    up = await rig.upload(P[:2])

    def delete() -> None:
        rig.db.docs.pop((rig.tenant, up.document_id))
        rig.db.versions.pop((rig.tenant, up.version_id))  # type: ignore[arg-type]

    rig.db.hooks["before_swap"] = delete
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert outcome.status is IngestStatus.SKIPPED


async def test_parse_failure_fails_the_version_for_good() -> None:
    rig = build(parser_error=DomainError("bad pdf", code="parse_failed"))
    up = await rig.upload(P[:2])

    with pytest.raises(IngestFailedError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up))

    assert info.value.code == "parse_failed"
    version = await rig.documents.version(rig.tenant, up.version_id)  # type: ignore[arg-type]
    assert version.status is VersionStatus.FAILED
    doc = await rig.documents.get(rig.tenant, rig.collection.id, up.document_id)
    assert doc.status is DocumentStatus.FAILED
    assert doc.error == {"code": "parse_failed", "message": "bad pdf"}
    [hook] = _webhook_jobs(rig)
    assert hook["event"] == "document.failed"
    assert hook["status"] == "failed"
    assert hook["error"] == {"code": "parse_failed", "message": "bad pdf"}
    assert rig.index.swaps == []


async def test_transient_embed_failure_is_retried_without_reparsing() -> None:
    rig = build()
    up = await rig.upload(P[:3])
    rig.embedders.error = UpstreamError("embedder 500")

    with pytest.raises(UpstreamError):
        await rig.ingest.ingest(rig.tenant, rig.target(up))

    version = await rig.documents.version(rig.tenant, up.version_id)  # type: ignore[arg-type]
    assert version.status is VersionStatus.PROCESSING
    doc = await rig.documents.get(rig.tenant, rig.collection.id, up.document_id)
    assert doc.status is DocumentStatus.EMBEDDING
    assert _webhook_jobs(rig) == []

    rig.embedders.error = None
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert outcome.status is IngestStatus.CURRENT
    assert rig.parser.calls == 1
    assert outcome.reused == 3


async def test_transient_failure_on_last_attempt_fails_the_version() -> None:
    rig = build()
    up = await rig.upload(P[:1])
    rig.embedders.error = UpstreamError("embedder 500")

    with pytest.raises(IngestFailedError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up), last_attempt=True)

    assert info.value.code == "upstream_failed"
    doc = await rig.documents.get(rig.tenant, rig.collection.id, up.document_id)
    assert doc.status is DocumentStatus.FAILED


async def test_unexpected_error_fails_with_internal_code() -> None:
    rig = build()
    up = await rig.upload(P[:1])
    rig.embedders.error = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    with pytest.raises(IngestFailedError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up), last_attempt=True)
    assert info.value.code == "internal"


async def test_wrong_vector_count_is_an_upstream_error() -> None:
    rig = build()
    up = await rig.upload(P[:2])
    rig.embedders.limit = 1
    with pytest.raises(UpstreamError, match="1 vectors for 2 texts"):
        await rig.ingest.ingest(rig.tenant, rig.target(up))


async def test_chunk_quota_fails_the_document() -> None:
    rig = build(quotas=Quotas(max_chunks=3))
    await rig.index_texts(P[:3])
    await rig.index_texts(P[3:6])
    assert rig.current_texts() == set(P[3:6])

    up = await rig.upload(P[:2], external_id="other")
    with pytest.raises(IngestFailedError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert info.value.code == "quota_exceeded"
    doc = await rig.documents.get(rig.tenant, rig.collection.id, up.document_id)
    assert doc.status is DocumentStatus.FAILED


async def test_unlimited_quota_skips_the_count() -> None:
    rig = build(quotas=Quotas())
    await rig.index_texts(P[:3])
    assert rig.current_texts() == set(P[:3])


@pytest.mark.parametrize(
    ("webhook", "events", "expected"),
    [
        (True, WebhookSettings(), ["document.ready"]),
        (True, WebhookSettings((WebhookEvent.DOCUMENT_FAILED,)), []),
        (False, WebhookSettings(), []),
    ],
)
async def test_ready_webhook_respects_subscription(
    webhook: bool, events: WebhookSettings, expected: list[str]
) -> None:
    rig = build(webhook=webhook, events=events)
    await rig.index_texts(P[:1])
    assert [h["event"] for h in _webhook_jobs(rig)] == expected


async def test_failed_webhook_respects_subscription() -> None:
    error = DomainError("bad", code="parse_failed")
    rig = build(events=WebhookSettings((WebhookEvent.DOCUMENT_READY,)), parser_error=error)
    up = await rig.upload(P[:1])
    with pytest.raises(IngestFailedError):
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert _webhook_jobs(rig) == []


async def test_ingest_dual_writes_during_reembed() -> None:
    rig = build()
    await rig.collections.set_target_space(rig.tenant, rig.collection.id, SPACE_B.id)
    await rig.index_texts(P[:3])

    current = rig.db.current_chunks(rig.tenant, rig.collection.id)
    for chunk_id in current:
        assert (rig.tenant, chunk_id, SPACE_A.id) in rig.db.vectors
        assert (rig.tenant, chunk_id, SPACE_B.id) in rig.db.vectors
    assert len(rig.embedders.texts(SPACE_A.id)) == len(rig.embedders.texts(SPACE_B.id)) == 3


async def test_reembed_started_before_swap_is_backfilled_then_swapped() -> None:
    rig = build()
    up = await rig.upload(P[:3])
    fired: list[int] = []

    def start_reembed() -> None:
        if not fired:
            fired.append(1)
            rig.db.col(rig.tenant, rig.collection.id).target_space_id = SPACE_B.id

    rig.db.hooks["before_swap"] = start_reembed
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))

    assert outcome.status is IngestStatus.CURRENT
    assert outcome.embedded == 6
    for chunk_id in rig.db.current_chunks(rig.tenant, rig.collection.id):
        assert (rig.tenant, chunk_id, SPACE_B.id) in rig.db.vectors


async def test_swap_gives_up_after_bounded_attempts() -> None:
    rig = build()
    up = await rig.upload(P[:1])

    def drop_vectors() -> None:
        rig.db.vectors.clear()

    rig.db.hooks["before_swap"] = drop_vectors
    with pytest.raises(ConflictError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert info.value.code == "vectors_missing"
    version = await rig.documents.version(rig.tenant, up.version_id)  # type: ignore[arg-type]
    assert version.status is VersionStatus.PROCESSING


async def test_other_conflicts_are_not_retried_in_place() -> None:
    rig = build()
    up = await rig.upload(P[:1])

    def conflict() -> None:
        raise ConflictError("lock timeout", code="busy")

    rig.db.hooks["before_swap"] = conflict
    with pytest.raises(ConflictError, match="lock timeout"):
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert rig.index.swaps == []


async def test_dimension_mismatch_fails_for_good() -> None:
    rig = build()
    up = await rig.upload(P[:1])
    rig.db.spaces[SPACE_A.id] = SPACE_A.__class__(SPACE_A.id, "space-a", "fake", "m", 4)
    with pytest.raises(IngestFailedError) as info:
        await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert info.value.code == "embedding_dimension_mismatch"


async def test_originals_dropped_when_not_retained() -> None:
    rig = build(retain_originals=False)
    up = await rig.index_texts(P[:2])
    assert (rig.tenant, up.version_id) not in rig.db.blobs

    kept = build(retain_originals=True)
    up = await kept.index_texts(P[:2])
    assert (kept.tenant, up.version_id) in kept.db.blobs


async def test_skipped_ready_version_still_drops_its_original() -> None:
    rig = build(retain_originals=False)
    up = await rig.upload(P[:2])
    await rig.ingest.ingest(rig.tenant, rig.target(up))
    rig.db.blobs[(rig.tenant, up.version_id)] = b"restored"  # type: ignore[index]
    outcome = await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert outcome.status is IngestStatus.SKIPPED
    assert (rig.tenant, up.version_id) not in rig.db.blobs


async def test_elements_already_stored_are_not_reparsed() -> None:
    rig = build()
    up = await rig.upload(["ignored"])
    elements = [Element(ElementKind.PARAGRAPH, P[4], 0)]
    await rig.documents.store_elements(rig.tenant, up.version_id, elements)  # type: ignore[arg-type]
    await rig.ingest.ingest(rig.tenant, rig.target(up))
    assert rig.parser.calls == 0
    assert rig.current_texts() == {P[4]}


def test_helpers() -> None:
    assert page_count([]) is None
    assert page_count([Element(ElementKind.PARAGRAPH, "x", 0, page=2, page_to=5)]) == 5
    assert error_body(ValidationError("f", "bad", code="c")) == {"code": "c", "message": "f: bad"}
    assert error_body(ValueError("x" * 600))["message"].startswith("ValueError: x")
    assert len(error_body(ValueError("x" * 600))["message"]) == 500
    assert is_transient(UpstreamError("x"))
    assert is_transient(ConflictError("x"))
    assert is_transient(RuntimeError("x"))
    assert not is_transient(QuotaExceededError("x"))
    assert not is_transient(ValidationError("f", "m"))


async def test_membership_pages_follow_drafts() -> None:
    rig = build()
    up = await rig.index_texts(P[:2])
    members: list[Membership] = rig.db.members[(rig.tenant, up.version_id)]  # type: ignore[index]
    assert [(m.page_from, m.page_to) for m in members] == [(1, 1), (2, 2)]
    hashes = {rig.db.chunks[(rig.tenant, m.chunk_id)].content_hash for m in members}
    assert len(hashes) == 2
    assert all(len(h) == len(content_hash("x")) for h in hashes)


async def test_gc_grace_is_configurable() -> None:
    rig = build()
    await rig.index_texts(P[:1])
    assert rig.db.jobs_of(rig.tenant, "gc")[0].delay == timedelta(minutes=10)
