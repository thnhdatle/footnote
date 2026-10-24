# SPDX-License-Identifier: Apache-2.0
"""Readiness queries and the queue depth gauge source."""

from __future__ import annotations

from footnote.adapters.postgres.session import Database, fetch_all, fetch_one

MIN_SERVER_VERSION_NUM = 150000
MIN_PGVECTOR = (0, 8, 0)

_VERSIONS = """
SELECT current_setting('server_version_num')::int AS server_version_num,
       (SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector') AS pgvector
"""
_REVISION = "SELECT version_num FROM footnote.alembic_version"
_QUEUE_DEPTH = (
    "SELECT kind, count(*)::int AS n FROM footnote.jobs WHERE status = 'queued' GROUP BY kind"
)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


async def ping(db: Database) -> list[str]:
    async with db.global_tx() as conn:
        await conn.execute("SELECT 1")
    return []


async def version_problems(db: Database) -> list[str]:
    async with db.global_tx() as conn:
        row = await fetch_one(conn, _VERSIONS)
    if row is None:  # pragma: no cover - a SELECT without FROM always returns a row
        return ["cannot read server version"]
    problems: list[str] = []
    if int(row["server_version_num"]) < MIN_SERVER_VERSION_NUM:
        problems.append(f"PostgreSQL {row['server_version_num']} < 15 is not supported")
    extversion = row["pgvector"]
    if extversion is None:
        problems.append("extension vector is not installed")
    elif _version_tuple(str(extversion)) < MIN_PGVECTOR:
        problems.append(f"pgvector {extversion} < 0.8.0 is not supported")
    return problems


async def migration_problems(db: Database, head: str | None) -> list[str]:
    if head is None:
        return ["migration scripts not found; cannot compare the schema revision"]
    async with db.global_tx() as conn:
        rows = await fetch_all(conn, _REVISION)
    current = sorted(str(r["version_num"]) for r in rows)
    if current != [head]:
        shown = ", ".join(current) or "none"
        return [f"schema revision {shown} is not the migration head {head}; run footnote migrate"]
    return []


async def queue_depth(admin: Database) -> dict[str, int]:
    async with admin.global_tx() as conn:
        rows = await fetch_all(conn, _QUEUE_DEPTH)
    return {str(r["kind"]): int(r["n"]) for r in rows}
