# SPDX-License-Identifier: Apache-2.0
"""Embedding-space registry.

Registration and deletion run over the admin pool: `footnote_register_space` is a
SECURITY DEFINER function executable by footnote_admin only, and dropping a space's
partial HNSW index needs the table owner. footnote_app reads spaces (SELECT only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from psycopg import errors, sql

from footnote.adapters.postgres.codec import space_from_row
from footnote.adapters.postgres.session import Database, fetch_all, fetch_one, require
from footnote.domain.embedding import EmbeddingSpace, SpaceSpec, index_type_for, validate_space_name
from footnote.domain.errors import ConflictError, ValidationError

_LOCK = "SELECT pg_advisory_xact_lock(hashtext('footnote_register_space'))"
_SPACE_COLS = (
    "SELECT id, name, provider, model, dimensions, send_dimensions, normalise, "
    "calibrated_threshold FROM footnote.embedding_spaces"
)


@dataclass(frozen=True, slots=True)
class HnswParams:
    """`embedding.hnsw` build parameters of new partial indexes."""

    m: int = 16
    ef_construction: int = 64


class PgSpaceReader:
    """Reads registered spaces (either pool)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def list_spaces(self) -> list[EmbeddingSpace]:
        """All spaces ordered by id."""
        async with self._db.global_tx() as conn:
            rows = await fetch_all(conn, _SPACE_COLS + " ORDER BY id")
        return [space_from_row(r) for r in rows]

    async def list_spaces_page(self, after: int | None, limit: int) -> list[EmbeddingSpace]:
        """Spaces ordered by id after `after` (keyset pagination)."""
        async with self._db.global_tx() as conn:
            rows = await fetch_all(
                conn,
                _SPACE_COLS + " WHERE %(after)s::bigint IS NULL OR id > %(after)s "
                "ORDER BY id LIMIT %(limit)s",
                {"after": after, "limit": limit},
            )
        return [space_from_row(r) for r in rows]

    async def get(self, name: str) -> EmbeddingSpace:
        """A space by name; NotFoundError when absent."""
        async with self._db.global_tx() as conn:
            row = await fetch_one(conn, _SPACE_COLS + " WHERE name = %(n)s", {"n": name})
        return space_from_row(require(row, f"embedding space {name!r} not found"))

    async def get_by_id(self, space_id: int) -> EmbeddingSpace:
        """A space by id; NotFoundError when absent."""
        async with self._db.global_tx() as conn:
            row = await fetch_one(conn, _SPACE_COLS + " WHERE id = %(i)s", {"i": space_id})
        return space_from_row(require(row, f"embedding space {space_id} not found"))


class PgSpaceAdmin(PgSpaceReader):
    """Registers and deletes spaces over the admin pool."""

    def __init__(self, admin: Database, hnsw: HnswParams, max_spaces: int = 8) -> None:
        super().__init__(admin)
        self._hnsw = hnsw
        self._max_spaces = max_spaces

    async def register(self, spec: SpaceSpec) -> EmbeddingSpace:
        """Idempotent by name; only a non-null threshold updates an existing row."""
        validate_space_name(spec.name)
        index_type_for(spec.dimensions)
        async with self._db.global_tx() as conn:
            await conn.execute(_LOCK)
            existing = await fetch_one(
                conn,
                "SELECT count(*) AS n, bool_or(name = %(n)s) AS present "
                "FROM footnote.embedding_spaces",
                {"n": spec.name},
            )
            assert existing is not None  # noqa: S101 - aggregate always returns a row
            if not existing["present"] and existing["n"] >= self._max_spaces:
                raise ConflictError(
                    f"at most {self._max_spaces} embedding spaces", code="space_limit_reached"
                )
            try:
                await conn.execute(
                    "SELECT footnote.footnote_register_space(%(name)s, %(provider)s, %(model)s, "
                    "%(dims)s, %(send)s, %(norm)s, %(thr)s, %(m)s, %(efc)s)",
                    {
                        "name": spec.name,
                        "provider": spec.provider,
                        "model": spec.model,
                        "dims": spec.dimensions,
                        "send": spec.send_dimensions,
                        "norm": spec.normalise,
                        "thr": spec.calibrated_threshold,
                        "m": self._hnsw.m,
                        "efc": self._hnsw.ef_construction,
                    },
                )
            except errors.InvalidParameterValue as exc:
                code, _, message = str(exc.diag.message_primary or exc).partition(": ")
                raise ValidationError("embedding_space", message, code=code) from exc
            row = await fetch_one(conn, _SPACE_COLS + " WHERE name = %(n)s", {"n": spec.name})
        return space_from_row(require(row, "space not registered"))

    async def delete(self, name: str) -> None:
        """Drop the space's index and row; 409 `space_in_use` while a collection uses it."""
        async with self._db.global_tx() as conn:
            await conn.execute(_LOCK)
            row = await fetch_one(
                conn,
                _SPACE_COLS + " WHERE name = %(n)s FOR UPDATE",
                {"n": name},
            )
            space = space_from_row(require(row, f"embedding space {name!r} not found"))
            used = await fetch_one(
                conn,
                "SELECT 1 FROM footnote.collections "
                "WHERE active_space_id = %(id)s OR target_space_id = %(id)s LIMIT 1",
                {"id": space.id},
            )
            if used is not None:
                raise ConflictError(f"embedding space {name!r} is in use", code="space_in_use")
            await conn.execute(
                sql.SQL("DROP INDEX IF EXISTS footnote.{}").format(
                    sql.Identifier(space.index_name())
                )
            )
            await conn.execute(
                "DELETE FROM footnote.embedding_spaces WHERE id = %(id)s", {"id": space.id}
            )


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_spacereader(db: Database) -> ports.SpaceReader:
        return PgSpaceReader(db)

    def _conforms_spaceregistry(db: Database) -> ports.SpaceRegistry:
        return PgSpaceAdmin(db, HnswParams())
