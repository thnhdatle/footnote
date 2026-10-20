# SPDX-License-Identifier: Apache-2.0
"""Re-embedding: backfill, dual-write with concurrent ingest, one atomic switch."""

from __future__ import annotations

from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from footnote.app.reembed import ReembedConfig, ReembedStatus, gc_space_job
from footnote.domain.documents import ChunkId, CollectionId, Membership
from footnote.domain.errors import ConflictError, NotFoundError, UpstreamError

from .rig import GC_GRACE, SPACE_A, SPACE_B, Rig, build, paragraph

P = [paragraph(f"w{n}x") for n in range(8)]


async def _start(rig: Rig) -> int:
    job_id = await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_B.name)
    assert job_id is not None
    record = await rig.jobs.get(rig.tenant, job_id)
    assert record.kind == "reembed"
    assert record.args == {"collection_id": str(rig.collection.id), "target_space_id": SPACE_B.id}
    return SPACE_B.id


def _all_current_have(rig: Rig, space_id: int) -> bool:
    return all(
        (rig.tenant, c, space_id) in rig.db.vectors
        for c in rig.db.current_chunks(rig.tenant, rig.collection.id)
    )


async def test_reembed_backfills_then_switches_once() -> None:
    rig = build()
    await rig.index_texts(P[:3], external_id="d1")
    await rig.index_texts(P[3:5], external_id="d2")
    target = await _start(rig)
    progress: list[tuple[int, int]] = []

    async def record(done: int, total: int) -> None:
        progress.append((done, total))

    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target, record)

    assert outcome.status is ReembedStatus.SWITCHED
    assert outcome.embedded == 5
    col = rig.db.col(rig.tenant, rig.collection.id)
    assert (col.active_space_id, col.target_space_id) == (SPACE_B.id, None)
    assert col.reembed is not None
    assert col.reembed["state"] == "done"
    assert rig.index.switches == [rig.collection.id]
    assert _all_current_have(rig, SPACE_B.id)
    assert progress == [(2, 5), (4, 5), (5, 5)]
    gc = [j for j in rig.db.jobs_of(rig.tenant, "gc") if "space_id" in j.args]
    assert [(j.args, j.delay) for j in gc] == [
        ({"collection_id": str(rig.collection.id), "space_id": SPACE_A.id}, GC_GRACE)
    ]
    # A second run (retry of a finished job) is a no-op.
    again = await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert again.status is ReembedStatus.CANCELLED
    assert rig.index.switches == [rig.collection.id]


async def test_reembed_of_an_empty_collection_switches_at_once() -> None:
    rig = build()
    target = await _start(rig)
    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert outcome == outcome.__class__(ReembedStatus.SWITCHED, 0)


@settings(max_examples=30)
@given(
    uploads=st.lists(
        st.tuples(
            st.sampled_from(["d1", "d2", "d3"]),
            st.lists(st.sampled_from(range(8)), min_size=1, max_size=3),
        ),
        min_size=0,
        max_size=6,
    )
)
async def test_property_reembed_with_concurrent_ingest_never_mixes_spaces(
    uploads: list[tuple[str, list[int]]],
) -> None:
    """Ingests between re-embed batches dual-write; queries always see one complete space."""
    rig = build()
    await rig.index_texts(P[:3], external_id="d1")
    await rig.index_texts(P[3:6], external_id="d2")
    target = await _start(rig)
    pending = list(uploads)
    observed: list[int] = []

    async def concurrent_ingest(space_id: int) -> None:
        # Every search snapshot taken now reads one space, complete for current chunks.
        view = rig.db.search_view(rig.tenant, rig.collection.id)
        observed.extend(set(view.values()))
        if pending and space_id == target and rig.embedders.before_embed is not None:
            ext, picks = pending.pop(0)
            rig.embedders.before_embed = None  # no recursion while this ingest embeds
            try:
                up = await rig.upload([P[i] for i in picks], external_id=ext)
                if not up.unchanged:
                    await rig.ingest.ingest(rig.tenant, rig.target(up))
            finally:
                rig.embedders.before_embed = concurrent_ingest
            rig.db.search_view(rig.tenant, rig.collection.id)

    rig.embedders.before_embed = concurrent_ingest
    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target)

    assert outcome.status is ReembedStatus.SWITCHED
    assert len(pending) < len(uploads) or not uploads  # ingests really interleaved
    assert rig.index.switches == [rig.collection.id]
    assert set(observed) <= {SPACE_A.id}  # before the switch only the old space is read
    assert rig.db.col(rig.tenant, rig.collection.id).active_space_id == SPACE_B.id
    assert _all_current_have(rig, SPACE_B.id)
    assert set(rig.db.search_view(rig.tenant, rig.collection.id).values()) <= {SPACE_B.id}


async def test_setting_the_active_space_back_cancels() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    target = await _start(rig)
    assert await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_A.name) is None

    col = rig.db.col(rig.tenant, rig.collection.id)
    assert (col.active_space_id, col.target_space_id, col.reembed) == (SPACE_A.id, None, None)
    gc = [j.args for j in rig.db.jobs_of(rig.tenant, "gc") if "space_id" in j.args]
    assert gc == [{"collection_id": str(rig.collection.id), "space_id": SPACE_B.id}]
    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert outcome.status is ReembedStatus.CANCELLED
    assert rig.index.switches == []


async def test_choosing_the_current_target_again_changes_nothing() -> None:
    rig = build()
    await _start(rig)
    assert await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_B.name) is None
    assert len(rig.db.jobs_of(rig.tenant, "reembed")) == 1


async def test_choosing_the_target_again_re_enqueues_a_lost_reembed_job() -> None:
    rig = build()
    target = await _start(rig)
    for job in rig.db.jobs_of(rig.tenant, "reembed"):
        job.status = "failed"  # e.g. the enqueue after set_target_space never committed
    job_id = await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_B.name)
    assert job_id is not None
    record = await rig.jobs.get(rig.tenant, job_id)
    assert record.args == {"collection_id": str(rig.collection.id), "target_space_id": target}


async def test_choosing_the_active_space_without_reembed_is_a_noop() -> None:
    rig = build()
    assert await rig.reembed.change_space(rig.tenant, rig.collection.id, SPACE_A.name) is None
    assert rig.db.jobs_of(rig.tenant) == []


async def test_changing_target_mid_reembed_schedules_gc_of_the_old_target() -> None:
    rig = build()
    space_c = rig.db.add_space(
        SPACE_B.__class__(3, "space-c", "fake", "model-c", SPACE_B.dimensions)
    )
    await _start(rig)
    job = await rig.reembed.change_space(rig.tenant, rig.collection.id, space_c.name)
    assert job is not None
    gc = [j.args for j in rig.db.jobs_of(rig.tenant, "gc")]
    assert gc == [{"collection_id": str(rig.collection.id), "space_id": SPACE_B.id}]


async def test_unknown_space_is_not_found() -> None:
    rig = build()
    with pytest.raises(NotFoundError):
        await rig.reembed.change_space(rig.tenant, rig.collection.id, "nope")


async def test_cancel_during_backfill_stops_without_switch() -> None:
    rig = build()
    await rig.index_texts(P[:5])
    target = await _start(rig)

    async def cancel(space_id: int) -> None:
        rig.db.col(rig.tenant, rig.collection.id).target_space_id = None

    rig.embedders.before_embed = cancel
    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert outcome.status is ReembedStatus.CANCELLED
    assert outcome.embedded == 2
    assert rig.index.switches == []
    assert rig.db.col(rig.tenant, rig.collection.id).active_space_id == SPACE_A.id


async def test_run_for_a_missing_collection_is_cancelled() -> None:
    rig = build()
    outcome = await rig.reembed.run(rig.tenant, CollectionId(UUID(int=5)), SPACE_B.id)
    assert outcome.status is ReembedStatus.CANCELLED


async def test_collection_deleted_during_run_is_cancelled() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    target = await _start(rig)

    async def delete(space_id: int) -> None:
        rig.db.col(rig.tenant, rig.collection.id).status = "deleting"

    rig.embedders.before_embed = delete
    outcome = await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert outcome.status is ReembedStatus.CANCELLED


async def test_wrong_vector_count_is_an_upstream_error() -> None:
    rig = build()
    await rig.index_texts(P[:3])
    target = await _start(rig)
    rig.embedders.limit = 1
    with pytest.raises(UpstreamError):
        await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert rig.db.col(rig.tenant, rig.collection.id).active_space_id == SPACE_A.id


async def test_switch_that_keeps_failing_raises_for_retry() -> None:
    rig = build(reembed_config=ReembedConfig(batch_size=2, max_switch_attempts=2))
    up = await rig.index_texts(P[:1])
    target = await _start(rig)
    counter = [0]

    def add_lacking_chunk() -> None:
        # A swap slipped in a current chunk without a target vector (cannot happen with
        # the real swap; simulates the race window between the batch query and switch).
        counter[0] += 1
        chunk = ChunkId(UUID(int=900 + counter[0]))
        first = next(iter(rig.db.chunks.values()))
        rig.db.chunks[(rig.tenant, chunk)] = first.__class__(
            chunk, rig.collection.id, up.document_id, f"{counter[0]:064x}", "text", first.language
        )
        rig.db.vectors[(rig.tenant, chunk, SPACE_A.id)] = [1.0] * 8
        key = (rig.tenant, up.version_id)
        rig.db.members[key].append(Membership(up.version_id, 99 + counter[0], chunk, None, None))  # type: ignore[index, arg-type]

    async def no_batch(*_: object) -> list[tuple[ChunkId, str]]:
        return []

    rig.index.reembed_batch = no_batch  # type: ignore[method-assign]
    rig.db.hooks["before_switch"] = add_lacking_chunk
    with pytest.raises(ConflictError) as info:
        await rig.reembed.run(rig.tenant, rig.collection.id, target)
    assert info.value.code == "reembed_incomplete"
    assert rig.index.switches == []


def test_gc_space_job_shape() -> None:
    job = gc_space_job(CollectionId(UUID(int=1)), 4, GC_GRACE)
    assert (job.kind, dict(job.args), job.delay) == (
        "gc",
        {"collection_id": str(UUID(int=1)), "space_id": 4},
        GC_GRACE,
    )
