# SPDX-License-Identifier: Apache-2.0
"""SECURITY DEFINER functions: key resolution, space registration, job claiming."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from footnote.adapters.postgres.jobs import PgJobs
from footnote.adapters.postgres.session import Database
from footnote.adapters.postgres.spaces import HnswParams, PgSpaceAdmin
from footnote.adapters.postgres.tenants import PgKeyResolver, PgTenantAdmin, hash_api_key
from footnote.domain.embedding import SpaceSpec
from footnote.domain.errors import (
    ConflictError,
    NotFoundError,
    UnauthenticatedError,
    ValidationError,
)
from footnote.domain.jobs import NewJob
from footnote.domain.tenancy import Quotas
from tests.integration.conftest import PgInfo
from tests.integration.world import seed

pytestmark = pytest.mark.integration


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _last_used(pg: PgInfo, key_id: uuid.UUID) -> datetime | None:
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT last_used_at FROM footnote.api_keys WHERE id = %s", (key_id,)
        )
        row = await cur.fetchone()
        assert row is not None
        value: datetime | None = row[0]
        return value


async def test_resolve_key_valid_and_last_used_throttled(
    pg: PgInfo, admin_db: Database, app_db: Database
) -> None:
    admin = PgTenantAdmin(admin_db)
    tenant = await admin.create_tenant(_name("keys"), Quotas())
    issued = await admin.create_key(tenant.id, "k")
    resolver = PgKeyResolver(app_db)

    assert await resolver.resolve(issued.key) == (tenant.id, issued.record.id)
    first = await _last_used(pg, issued.record.id)
    assert first is not None
    await resolver.resolve(issued.key)
    assert await _last_used(pg, issued.record.id) == first  # at most once per minute

    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        await conn.execute(
            "UPDATE footnote.api_keys SET last_used_at = now() - interval '2 minutes' "
            "WHERE id = %s",
            (issued.record.id,),
        )
    stale = await _last_used(pg, issued.record.id)
    await resolver.resolve(issued.key)
    refreshed = await _last_used(pg, issued.record.id)
    assert stale is not None
    assert refreshed is not None
    assert refreshed > stale


async def test_resolve_key_rejects_unknown_revoked_expired_and_deleting(
    admin_db: Database, app_db: Database
) -> None:
    admin = PgTenantAdmin(admin_db)
    resolver = PgKeyResolver(app_db)
    tenant = await admin.create_tenant(_name("keys"), Quotas())
    revoked = await admin.create_key(tenant.id, "revoked")
    await admin.revoke_key(tenant.id, revoked.record.id)
    expired = await admin.create_key(tenant.id, "expired", datetime.now(UTC) - timedelta(seconds=1))
    live = await admin.create_key(tenant.id, "live")
    for key in ("fn_unknown", "not-a-key", revoked.key, expired.key):
        with pytest.raises(UnauthenticatedError):
            await resolver.resolve(key)
    assert (await resolver.resolve(live.key))[0] == tenant.id
    await admin.mark_deleting(tenant.id)
    with pytest.raises(UnauthenticatedError):
        await resolver.resolve(live.key)


async def test_rotate_key_keeps_old_key_for_grace(admin_db: Database, app_db: Database) -> None:
    admin = PgTenantAdmin(admin_db)
    resolver = PgKeyResolver(app_db)
    tenant = await admin.create_tenant(_name("rotate"), Quotas())
    old = await admin.create_key(tenant.id, "ci")
    new = await admin.rotate_key(tenant.id, old.record.id, timedelta(hours=24))
    assert new.record.name == "ci"
    assert new.key != old.key
    assert (await resolver.resolve(old.key))[1] == old.record.id
    assert (await resolver.resolve(new.key))[1] == new.record.id
    newer = await admin.rotate_key(tenant.id, new.record.id, timedelta(0))
    with pytest.raises(UnauthenticatedError):
        await resolver.resolve(new.key)
    assert (await resolver.resolve(newer.key))[1] == newer.record.id


async def test_app_calls_resolve_function_directly_but_cannot_read_keys(
    pg: PgInfo, admin_db: Database
) -> None:
    admin = PgTenantAdmin(admin_db)
    tenant = await admin.create_tenant(_name("direct"), Quotas())
    issued = await admin.create_key(tenant.id, "k")
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT * FROM footnote.footnote_resolve_key(%s)", (hash_api_key(issued.key),)
        )
        assert await cur.fetchall() == [(tenant.id, issued.record.id)]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute("SELECT key_hash FROM footnote.api_keys")


async def test_register_space_is_idempotent_and_updates_only_threshold(admin_db: Database) -> None:
    spaces = PgSpaceAdmin(admin_db, HnswParams(m=8, ef_construction=32), max_spaces=1000)
    name = _name("sp")
    first = await spaces.register(SpaceSpec(name, "openai_compat", "m1", 16, False, True, 0.3))
    assert (first.dimensions, first.calibrated_threshold) == (16, 0.3)
    again = await spaces.register(SpaceSpec(name, "onnx_local", "m2", 32, True, False, None))
    assert again == first
    updated = await spaces.register(SpaceSpec(name, "x", "y", 99, False, True, 0.55))
    assert updated.id == first.id
    assert updated.dimensions == 16
    assert updated.calibrated_threshold == 0.55
    assert await spaces.get(name) == updated
    assert updated in await spaces.list_spaces()
    page = await spaces.list_spaces_page(first.id - 1, 1)
    assert page == [updated]
    assert updated not in await spaces.list_spaces_page(first.id, 100)


async def test_register_space_index_types_and_limits(pg: PgInfo, admin_db: Database) -> None:
    spaces = PgSpaceAdmin(admin_db, HnswParams(), max_spaces=1000)
    half = await spaces.register(SpaceSpec(_name("half"), "openai_compat", "big", 3072, True))
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s", (half.index_name(),)
        )
        row = await cur.fetchone()
        assert row is not None
        assert "((embedding)::halfvec(3072)) halfvec_cosine_ops" in row[0]
        assert f"WHERE (space_id = {half.id})" in row[0]
        with pytest.raises(psycopg.errors.InvalidParameterValue, match="embedding_dimension"):
            await conn.execute(
                "SELECT footnote.footnote_register_space('too-big', 'p', 'm', 4001, false, true, "
                "null)"
            )
    with pytest.raises(ValidationError) as err:
        await spaces.register(SpaceSpec(_name("huge"), "p", "m", 4001))
    assert err.value.code == "embedding_dimension_unsupported"
    with pytest.raises(ValidationError):
        await spaces.register(SpaceSpec("Bad Name", "p", "m", 8))

    current = len(await spaces.list_spaces())
    capped = PgSpaceAdmin(admin_db, HnswParams(), max_spaces=current)
    with pytest.raises(ConflictError) as limit:
        await capped.register(SpaceSpec(_name("over"), "p", "m", 8))
    assert limit.value.code == "space_limit_reached"
    assert (await capped.register(SpaceSpec(half.name, "p", "m", 3072))).id == half.id


async def test_wrong_dimension_vector_is_rejected_by_index_expression(
    pg: PgInfo, admin_db: Database, app_db: Database
) -> None:
    world = await seed(admin_db, app_db, _name("dims"), ["one chunk"])
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        await conn.execute(
            "SELECT set_config('footnote.tenant_id', %s, false)", (str(world.tenant),)
        )
        await conn.execute(
            "DELETE FROM footnote.chunk_vectors WHERE chunk_id = %s", (world.chunk_ids[0],)
        )
        with pytest.raises(psycopg.errors.DataException):
            await conn.execute(
                "INSERT INTO footnote.chunk_vectors (tenant_id, chunk_id, space_id, "
                "collection_id, embedding) VALUES (%s, %s, %s, %s, '[1,0,0,0]')",
                (world.tenant, world.chunk_ids[0], world.space.id, world.collection.id),
            )


async def test_delete_space_in_use_and_unused(
    pg: PgInfo, admin_db: Database, app_db: Database
) -> None:
    spaces = PgSpaceAdmin(admin_db, HnswParams(), max_spaces=1000)
    world = await seed(admin_db, app_db, _name("del"), ["text"])
    with pytest.raises(ConflictError) as used:
        await spaces.delete(world.space.name)
    assert used.value.code == "space_in_use"
    spare = await spaces.register(SpaceSpec(_name("spare"), "p", "m", 4))
    await spaces.delete(spare.name)
    with pytest.raises(NotFoundError):
        await spaces.get(spare.name)
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM pg_indexes WHERE indexname = %s", (spare.index_name(),)
        )
        assert await cur.fetchone() == (0,)
    with pytest.raises(NotFoundError):
        await spaces.delete(spare.name)


async def test_claim_jobs_across_tenants_skip_locked_and_lease_expiry(
    pg: PgInfo, admin_db: Database, app_db: Database
) -> None:
    admin = PgTenantAdmin(admin_db)
    jobs = PgJobs(app_db)
    t1 = (await admin.create_tenant(_name("q1"), Quotas())).id
    t2 = (await admin.create_tenant(_name("q2"), Quotas())).id
    j1 = await jobs.enqueue(t1, NewJob("webhook", {"n": 1}))
    j2 = await jobs.enqueue(t2, NewJob("webhook", {"n": 2}))
    later = await jobs.enqueue(t1, NewJob("gc", {}, timedelta(hours=1)))

    first, second = await asyncio.gather(
        jobs.claim("w1", 1000, timedelta(minutes=5)), jobs.claim("w2", 1000, timedelta(minutes=5))
    )
    ids_1 = {c.id for c in first}
    ids_2 = {c.id for c in second}
    assert not ids_1 & ids_2
    claimed = {c.id: c for c in first + second}
    assert {j1, j2} <= set(claimed)
    assert later not in claimed
    assert (claimed[j1].tenant_id, claimed[j1].kind) == (t1, "webhook")
    owner = "w1" if j1 in ids_1 else "w2"
    other = "w2" if owner == "w1" else "w1"
    record = await jobs.get(t1, j1)
    assert (record.status, record.attempts) == ("running", 1)

    # Expired lease: reclaimable by another worker; the old owner can no longer write.
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        await conn.execute(
            "UPDATE footnote.jobs SET lease_until = now() - interval '1 second' WHERE id = %s",
            (j1,),
        )
    reclaimed = await jobs.claim(other, 1000, timedelta(minutes=5))
    assert j1 in {c.id for c in reclaimed}
    assert not await jobs.complete(t1, j1, owner)
    assert await jobs.extend_lease(t1, j1, other, timedelta(minutes=10))
    assert await jobs.set_progress(t1, j1, other, 3, 10)
    assert await jobs.complete(t1, j1, other)
    done = await jobs.get(t1, j1)
    assert (done.status, done.attempts, done.progress) == ("done", 2, {"done": 3, "total": 10})
    assert done.finished_at is not None


async def test_claim_jobs_validates_arguments(pg: PgInfo) -> None:
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        for args in (
            ("w", 0, "1 minute"),
            ("w", 1001, "1 minute"),
            ("", 1, "1 minute"),
            ("w", 1, "0 seconds"),
        ):
            with pytest.raises(psycopg.errors.InvalidParameterValue):
                await conn.execute(
                    "SELECT * FROM footnote.footnote_claim_jobs(%s, %s, %s::interval)", args
                )
