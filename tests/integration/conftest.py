# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from testcontainers.community.postgres import PostgresContainer

from footnote.adapters.postgres.session import Database, Pool, PoolSettings, open_pool
from tests.integration.world import DIMS, SPACE_NAME

ROOT = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("FOOTNOTE_TEST_PG_IMAGE", "pgvector/pgvector:0.8.1-pg17")
ADMIN_PASSWORD = "admin-pw"
APP_PASSWORD = "app-pw"


@dataclass(frozen=True)
class PgInfo:
    superuser: str
    admin: str
    app: str
    space_id: int


def _dsn(container: PostgresContainer, user: str, password: str) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    return f"postgresql://{user}:{password}@{host}:{port}/footnote"


def migrate(admin_dsn: str) -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("path_separator", "os")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FOOTNOTE_ADMIN_DATABASE_URL", admin_dsn)
        command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def pg() -> Iterator[PgInfo]:
    container = PostgresContainer(
        IMAGE, username="postgres", password="postgres", dbname="footnote", driver=None
    )
    container.with_volume_mapping(str(ROOT / "deploy" / "sql"), "/bootstrap", "ro")
    _start(container)
    try:
        code, output = container.exec(
            [
                "psql",
                "-U",
                "postgres",
                "-d",
                "footnote",
                "-v",
                "ON_ERROR_STOP=1",
                "-v",
                f"app_password='{APP_PASSWORD}'",
                "-v",
                f"admin_password='{ADMIN_PASSWORD}'",
                "-f",
                "/bootstrap/bootstrap.sql",
            ]
        )
        assert code == 0, output.decode()
        admin = _dsn(container, "footnote_admin", ADMIN_PASSWORD)
        migrate(admin)
        with psycopg.connect(admin, autocommit=True) as conn:
            row = conn.execute(
                "SELECT footnote.footnote_register_space(%s, 'fake', 'fake-model', %s, false, "
                "true, 0.4)",
                (SPACE_NAME, DIMS),
            ).fetchone()
        assert row is not None
        yield PgInfo(
            superuser=_dsn(container, "postgres", "postgres"),
            admin=admin,
            app=_dsn(container, "footnote_app", APP_PASSWORD),
            space_id=int(row[0]),
        )
    finally:
        container.stop()


def _start(container: PostgresContainer, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            container.start()
        except ConnectionError:
            if attempt == attempts:
                raise
        else:
            return


async def _pool(dsn: str) -> Pool:
    return await open_pool(PoolSettings(conninfo=dsn, min_size=1, max_size=4))


@pytest_asyncio.fixture
async def app_pool(pg: PgInfo) -> AsyncIterator[Pool]:
    pool = await _pool(pg.app)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def admin_pool(pg: PgInfo) -> AsyncIterator[Pool]:
    pool = await _pool(pg.admin)
    yield pool
    await pool.close()


@pytest.fixture
def app_db(app_pool: Pool) -> Database:
    return Database(app_pool)


@pytest.fixture
def admin_db(admin_pool: Pool) -> Database:
    return Database(admin_pool)


@pytest.fixture(autouse=True)
def _clean_tenants_and_spaces(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "pg" not in request.fixturenames or "filler" in request.fixturenames:
        return
    pg: PgInfo = request.getfixturevalue("pg")
    drop_extra_data(pg)


def drop_extra_data(pg: PgInfo) -> None:
    with psycopg.connect(pg.admin, autocommit=True) as conn:
        conn.execute("DELETE FROM footnote.jobs")
        conn.execute("DELETE FROM footnote.tenants")
        rows = conn.execute(
            "SELECT id FROM footnote.embedding_spaces WHERE id <> %s", (pg.space_id,)
        ).fetchall()
        for (space_id,) in rows:
            conn.execute(f"DROP INDEX IF EXISTS footnote.chunk_vectors_hnsw_s{int(space_id)}")
            conn.execute("DELETE FROM footnote.embedding_spaces WHERE id = %s", (space_id,))
