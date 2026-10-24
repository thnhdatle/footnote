# SPDX-License-Identifier: Apache-2.0
"""Connection pools and tenant-scoped transactions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import orjson
from psycopg import AsyncConnection, errors, sql
from psycopg.rows import TupleRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from ulid import ULID

from footnote.domain.errors import ConflictError, NotFoundError, ValidationError
from footnote.domain.tenancy import TenantId

type Conn = AsyncConnection[TupleRow]
type Pool = AsyncConnectionPool[AsyncConnection[TupleRow]]
type Row = dict[str, Any]
type Params = Mapping[str, Any]
type Query = str | bytes | sql.SQL | sql.Composed

_SET_TENANT = (
    "SELECT set_config('footnote.tenant_id', %(tenant)s, true), "
    "set_config('statement_timeout', %(timeout)s, true)"
)
_SET_TIMEOUT = "SELECT set_config('statement_timeout', %(timeout)s, true)"


def new_id() -> UUID:
    return ULID().to_uuid()


def jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda v: orjson.dumps(v).decode())


@dataclass(frozen=True, slots=True)
class PoolSettings:
    conninfo: str
    min_size: int = 1
    max_size: int = 20
    prepare_threshold: int | None = 5
    open_timeout: float = 30.0


async def open_pool(settings: PoolSettings) -> Pool:
    """Opens an async connection pool."""
    pool: Pool = AsyncConnectionPool(
        settings.conninfo,
        min_size=settings.min_size,
        max_size=settings.max_size,
        kwargs={"autocommit": True, "prepare_threshold": settings.prepare_threshold},
        check=AsyncConnectionPool.check_connection,
        open=False,
        name="footnote",
    )
    await pool.open(wait=True, timeout=settings.open_timeout)
    return pool


@dataclass(frozen=True, slots=True)
class Database:
    """A pool with the transaction policy shared by the repositories."""

    pool: Pool
    statement_timeout_ms: int = 30_000
    app_tenant_predicates: bool = True

    def tenant_filter(self, alias: str) -> sql.Composable:
        if not self.app_tenant_predicates:
            return sql.SQL("TRUE")
        return sql.SQL("{}.tenant_id = %(tenant)s").format(sql.Identifier(alias))

    @asynccontextmanager
    async def tenant_tx(
        self, tenant: TenantId, *, read_only_snapshot: bool = False
    ) -> AsyncIterator[Conn]:
        """Opens a transaction scoped to a tenant by row-level security."""
        async with self.pool.connection() as conn, translate_errors(), conn.transaction():
            if read_only_snapshot:
                await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            await conn.execute(
                _SET_TENANT, {"tenant": str(tenant), "timeout": f"{self.statement_timeout_ms}"}
            )
            yield conn

    @asynccontextmanager
    async def global_tx(self) -> AsyncIterator[Conn]:
        """Opens a transaction without a tenant."""
        async with self.pool.connection() as conn, translate_errors(), conn.transaction():
            await conn.execute(_SET_TIMEOUT, {"timeout": f"{self.statement_timeout_ms}"})
            yield conn


@asynccontextmanager
async def translate_errors() -> AsyncIterator[None]:
    try:
        yield
    except errors.UniqueViolation as exc:
        raise ConflictError(f"already exists ({_constraint(exc)})") from exc
    except errors.ForeignKeyViolation as exc:
        raise NotFoundError(f"referenced resource not found ({_constraint(exc)})") from exc
    except errors.CheckViolation as exc:
        raise ValidationError(_constraint(exc), "value out of range") from exc


def _constraint(exc: errors.Error) -> str:
    return exc.diag.constraint_name or "constraint"


async def fetch_all(conn: Conn, query: Query, params: Params | None = None) -> list[Row]:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        return await cur.fetchall()


async def fetch_one(conn: Conn, query: Query, params: Params | None = None) -> Row | None:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        return await cur.fetchone()


async def execute(conn: Conn, query: Query, params: Params | None = None) -> int:
    async with conn.cursor() as cur:
        await cur.execute(query, params)
        return cur.rowcount


def require[T](value: T | None, message: str) -> T:
    if value is None:
        raise NotFoundError(message)
    return value


type IdFactory = Callable[[], UUID]
