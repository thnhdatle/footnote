# SPDX-License-Identifier: Apache-2.0
"""Raw-SQL RLS isolation as footnote_app."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import psycopg
import pytest
import pytest_asyncio

from footnote.adapters.postgres.session import Database
from tests.integration.conftest import PgInfo
from tests.integration.world import World, seed

pytestmark = pytest.mark.integration

APP_TABLES = (
    "collections",
    "documents",
    "document_versions",
    "document_blobs",
    "chunks",
    "version_chunks",
    "chunk_vectors",
    "jobs",
    "webhook_deliveries",
    "tenant_usage",
)


@pytest_asyncio.fixture
async def two(admin_db: Database, app_db: Database) -> AsyncIterator[tuple[World, World]]:
    """Tenants A and B, each with rows in every tenant table."""
    a = await seed(admin_db, app_db, "rls-a", ["alpha one", "alpha two"])
    b = await seed(admin_db, app_db, "rls-b", ["bravo one", "bravo two", "bravo three"])
    yield a, b


async def _count_all(dsn: str, setting: str | None, table: str) -> tuple[int, int]:
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        if setting is not None:
            await conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (setting,))
        cur = await conn.execute(f"SELECT count(*) FROM footnote.{table}")  # noqa: S608 - constant table names
        total = (await cur.fetchone() or (0,))[0]
        return total, 0


async def _admin_count(pg: PgInfo, table: str, tenant: uuid.UUID) -> int:
    async with await psycopg.AsyncConnection.connect(pg.admin, autocommit=True) as conn:
        cur = await conn.execute(
            f"SELECT count(*) FROM footnote.{table} WHERE tenant_id = %s",  # noqa: S608 - constant
            (tenant,),
        )
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])


@pytest.mark.parametrize("setting", [None, "", "random"])
async def test_no_rows_without_a_valid_tenant(
    pg: PgInfo, two: tuple[World, World], setting: str | None
) -> None:
    value = str(uuid.uuid4()) if setting == "random" else setting
    for table in APP_TABLES:
        total, _ = await _count_all(pg.app, value, table)
        assert total == 0, (table, setting)


async def test_other_tenant_setting_sees_only_its_own_rows(
    pg: PgInfo, two: tuple[World, World]
) -> None:
    a, b = two
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        await conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (str(b.tenant),))
        for table in APP_TABLES:
            cur = await conn.execute(
                f"SELECT count(*) FILTER (WHERE tenant_id = %s), "  # noqa: S608 - constant
                f"count(*) FILTER (WHERE tenant_id <> %s) FROM footnote.{table}",
                (b.tenant, b.tenant),
            )
            own, foreign = await cur.fetchone() or (0, 0)
            assert foreign == 0, table
            assert own == await _admin_count(pg, table, b.tenant), table
            assert await _admin_count(pg, table, a.tenant) > 0, table


async def test_cross_tenant_updates_and_deletes_touch_nothing(
    pg: PgInfo, two: tuple[World, World]
) -> None:
    a, b = two
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        await conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (str(b.tenant),))
        for table in APP_TABLES:
            cur = await conn.execute(
                f"UPDATE footnote.{table} SET tenant_id = tenant_id WHERE tenant_id = %s",  # noqa: S608
                (a.tenant,),
            )
            assert cur.rowcount == 0, table
            cur = await conn.execute(
                f"DELETE FROM footnote.{table} WHERE tenant_id = %s",  # noqa: S608
                (a.tenant,),
            )
            assert cur.rowcount == 0, table
    for table in APP_TABLES:
        assert await _admin_count(pg, table, a.tenant) > 0, table


async def test_insert_referencing_another_tenants_ids_fails(
    pg: PgInfo, two: tuple[World, World]
) -> None:
    a, b = two
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        await conn.execute("SELECT set_config('footnote.tenant_id', %s, false)", (str(b.tenant),))
        # Own tenant id, foreign parent id: the composite FK finds no (B, A-id) parent.
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await conn.execute(
                "INSERT INTO footnote.documents (tenant_id, id, collection_id, external_id) "
                "VALUES (%s, gen_random_uuid(), %s, 'x')",
                (b.tenant, a.collection.id),
            )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await conn.execute(
                "INSERT INTO footnote.version_chunks (tenant_id, version_id, ordinal, chunk_id) "
                "VALUES (%s, %s, 99, %s)",
                (b.tenant, b.version_id, a.chunk_ids[0]),
            )
        # Foreign tenant id: WITH CHECK rejects the row.
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute(
                "INSERT INTO footnote.jobs (tenant_id, id, kind) "
                "VALUES (%s, gen_random_uuid(), 'gc')",
                (a.tenant,),
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute(
                "UPDATE footnote.jobs SET tenant_id = %s WHERE tenant_id = %s",
                (a.tenant, b.tenant),
            )


async def test_malformed_setting_is_an_error_not_a_bypass(
    pg: PgInfo, two: tuple[World, World]
) -> None:
    async with await psycopg.AsyncConnection.connect(pg.app, autocommit=True) as conn:
        await conn.execute("SELECT set_config('footnote.tenant_id', 'not-a-uuid', false)")
        with pytest.raises(psycopg.errors.InvalidTextRepresentation):
            await conn.execute("SELECT count(*) FROM footnote.chunks")


async def test_tenant_setting_is_transaction_local(
    app_db: Database, two: tuple[World, World]
) -> None:
    a, _ = two
    async with app_db.tenant_tx(a.tenant) as conn:
        cur = await conn.execute("SELECT current_setting('footnote.tenant_id', true)")
        assert await cur.fetchone() == (str(a.tenant),)
    async with app_db.pool.connection() as conn:
        cur = await conn.execute("SELECT current_setting('footnote.tenant_id', true)")
        row = await cur.fetchone()
        assert row is not None
        assert row[0] in (None, "")
        cur = await conn.execute("SELECT count(*) FROM footnote.chunks")
        assert await cur.fetchone() == (0,)


async def test_fault_mode_without_app_predicates_still_isolates(
    app_db: Database, two: tuple[World, World]
) -> None:
    a, b = two
    fault = Database(app_db.pool, app_tenant_predicates=False)
    async with fault.tenant_tx(b.tenant) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM footnote.chunks WHERE tenant_id = %s", (a.tenant,)
        )
        assert await cur.fetchone() == (0,)
