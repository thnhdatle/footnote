# SPDX-License-Identifier: Apache-2.0
"""Connection pools and tenant-scoped transactions (row-level security per tenant).

Every tenant transaction starts with `set_config('footnote.tenant_id', $1, true)`; the
tenant id is always a bind parameter, never interpolated into SQL text. The setting is
transaction-local, so pooled connections (and transaction poolers) never carry a tenant
from one transaction to the next.
"""

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
    """A new ULID stored as uuid."""
    return ULID().to_uuid()


def jsonb(value: Any) -> Jsonb:
    """Wrap a JSON-ready value (UUIDs and datetimes allowed) for a jsonb parameter."""
    return Jsonb(value, dumps=lambda v: orjson.dumps(v).decode())


@dataclass(frozen=True, slots=True)
class PoolSettings:
    """Pool options from config (`database.*` / `admin_database.*`).

    `prepare_threshold` None disables psycopg auto-prepare (needed behind a transaction
    pooler).
    """

    conninfo: str
    min_size: int = 1
    max_size: int = 20
    prepare_threshold: int | None = 5
    open_timeout: float = 30.0


async def open_pool(settings: PoolSettings) -> Pool:
    """Open an async pool of autocommit connections; transactions are always explicit."""
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
    """A pool plus the per-transaction policy shared by the repositories.

    `app_tenant_predicates` False removes the defence-in-depth `tenant_id = %(tenant)s`
    predicates (tenant-isolation fault test only; RLS must still isolate tenants).
    """

    pool: Pool
    statement_timeout_ms: int = 30_000
    app_tenant_predicates: bool = True

    def tenant_filter(self, alias: str) -> sql.Composable:
        """`<alias>.tenant_id = %(tenant)s`, or TRUE in the tenant-isolation fault test."""
        if not self.app_tenant_predicates:
            return sql.SQL("TRUE")
        return sql.SQL("{}.tenant_id = %(tenant)s").format(sql.Identifier(alias))

    @asynccontextmanager
    async def tenant_tx(
        self, tenant: TenantId, *, read_only_snapshot: bool = False
    ) -> AsyncIterator[Conn]:
        """Transaction scoped to `tenant` by the RLS setting; commits on success."""
        async with self.pool.connection() as conn, translate_errors(), conn.transaction():
            if read_only_snapshot:
                await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            await conn.execute(
                _SET_TENANT, {"tenant": str(tenant), "timeout": f"{self.statement_timeout_ms}"}
            )
            yield conn

    @asynccontextmanager
    async def global_tx(self) -> AsyncIterator[Conn]:
        """Transaction without a tenant (admin pool, definer-function calls)."""
        async with self.pool.connection() as conn, translate_errors(), conn.transaction():
            await conn.execute(_SET_TIMEOUT, {"timeout": f"{self.statement_timeout_ms}"})
            yield conn


@asynccontextmanager
async def translate_errors() -> AsyncIterator[None]:
    """Map integrity errors to domain errors at the adapter boundary."""
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
    """Run `query` and return every row as a dict."""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        return await cur.fetchall()


async def fetch_one(conn: Conn, query: Query, params: Params | None = None) -> Row | None:
    """Run `query` and return the first row as a dict, or None."""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        return await cur.fetchone()


async def execute(conn: Conn, query: Query, params: Params | None = None) -> int:
    """Run a statement and return its row count."""
    async with conn.cursor() as cur:
        await cur.execute(query, params)
        return cur.rowcount


def require[T](value: T | None, message: str) -> T:
    """Return `value` or raise NotFoundError (other tenants' ids look absent too)."""
    if value is None:
        raise NotFoundError(message)
    return value


type IdFactory = Callable[[], UUID]
