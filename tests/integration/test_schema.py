# SPDX-License-Identifier: Apache-2.0
"""Migrations, catalog and role assertions."""

from __future__ import annotations

import psycopg
import pytest
from psycopg import sql

from footnote.adapters.postgres.roles import app_role_problems
from footnote.adapters.postgres.session import Database
from tests.integration.conftest import PgInfo, migrate

pytestmark = pytest.mark.integration

TENANT_TABLES = {
    "api_keys",
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
}
APP_DML_TABLES = TENANT_TABLES - {"api_keys"}
DEFINER_FUNCTIONS = {
    "footnote_resolve_key": {"footnote_admin", "footnote_app"},
    "footnote_claim_jobs": {"footnote_admin", "footnote_app"},
    "footnote_register_space": {"footnote_admin"},
}


def _rows(dsn: str, query: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute(query, params).fetchall()


def test_migration_is_at_head_and_reapplies_as_noop(pg: PgInfo) -> None:
    migrate(pg.admin)
    assert _rows(pg.app, "SELECT version_num FROM footnote.alembic_version") == [("0001",)]


def test_every_object_in_schema_is_owned_by_admin(pg: PgInfo) -> None:
    owners = _rows(
        pg.superuser,
        "SELECT DISTINCT pg_get_userbyid(relowner) FROM pg_class "
        "WHERE relnamespace = 'footnote'::regnamespace "
        "UNION SELECT DISTINCT pg_get_userbyid(proowner) FROM pg_proc "
        "WHERE pronamespace = 'footnote'::regnamespace "
        "UNION SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = 'footnote'",
    )
    assert owners == [("footnote_admin",)]


def test_rls_enabled_and_forced_with_both_policies(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, "
        "array_agg(p.polname ORDER BY p.polname) FILTER (WHERE p.polname IS NOT NULL) "
        "FROM pg_class c LEFT JOIN pg_policy p ON p.polrelid = c.oid "
        "WHERE c.relnamespace = 'footnote'::regnamespace AND c.relkind = 'r' "
        "GROUP BY c.relname, c.relrowsecurity, c.relforcerowsecurity",
    )
    by_table = {r[0]: r[1:] for r in rows}
    for table in TENANT_TABLES:
        assert by_table[table] == (True, True, ["admin_all", "tenant_isolation"]), table
    assert by_table["tenants"] == (True, True, ["admin_all"])
    policies = _rows(
        pg.superuser,
        "SELECT tablename, policyname, roles::text[], qual, with_check FROM pg_policies "
        "WHERE schemaname = 'footnote'",
    )
    for table, name, roles, qual, check in policies:
        if name == "tenant_isolation":
            assert roles == ["footnote_app"], table
            expected = (
                "(tenant_id = (NULLIF(current_setting('footnote.tenant_id'::text, true), "
                "''::text))::uuid)"
            )
            assert qual == expected
            assert check == expected
        else:
            assert (name, roles, qual, check) == ("admin_all", ["footnote_admin"], "true", "true")


def test_app_role_grants(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT table_name, string_agg(privilege_type, ',' ORDER BY privilege_type) "
        "FROM information_schema.role_table_grants "
        "WHERE grantee = 'footnote_app' AND table_schema = 'footnote' GROUP BY table_name",
    )
    grants = dict(rows)
    for table in APP_DML_TABLES:
        assert grants.pop(table) == "DELETE,INSERT,SELECT,UPDATE", table
    assert grants == {"embedding_spaces": "SELECT", "alembic_version": "SELECT"}


@pytest.mark.parametrize("table", ["tenants", "api_keys"])
def test_app_is_denied_on_tenants_and_keys(pg: PgInfo, table: str) -> None:
    with psycopg.connect(pg.app, autocommit=True) as conn:
        ident = sql.Identifier("footnote", table)
        for stmt in (
            sql.SQL("SELECT * FROM {}").format(ident),
            sql.SQL("DELETE FROM {}").format(ident),
            sql.SQL("UPDATE {} SET id = id").format(ident),
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(stmt)


def test_app_cannot_create_objects_or_write_spaces(pg: PgInfo) -> None:
    with psycopg.connect(pg.app, autocommit=True) as conn:
        for stmt in (
            "CREATE INDEX x ON footnote.chunk_vectors (space_id)",
            "CREATE TABLE footnote.x (id int)",
            "INSERT INTO footnote.embedding_spaces (name, provider, model, dimensions, "
            "index_type) VALUES ('x', 'p', 'm', 4, 'vector')",
            "SELECT footnote.footnote_register_space('x', 'p', 'm', 4, false, true, null)",
            "ALTER TABLE footnote.chunks DISABLE ROW LEVEL SECURITY",
            "SET ROLE footnote_admin",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(stmt)


def test_definer_functions_are_hardened(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT p.proname, pg_get_userbyid(p.proowner), p.prosecdef, p.proconfig, "
        "ARRAY(SELECT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END "
        "      FROM aclexplode(p.proacl) a WHERE a.privilege_type = 'EXECUTE') "
        "FROM pg_proc p WHERE p.pronamespace = 'footnote'::regnamespace",
    )
    seen = {}
    for name, owner, secdef, config, grantees in rows:
        assert owner == "footnote_admin", name
        assert secdef is True, name
        assert config == ["search_path=pg_catalog, pg_temp"], name
        assert "PUBLIC" not in grantees, name
        seen[name] = set(grantees)
    assert seen == DEFINER_FUNCTIONS


def test_roles_have_no_superuser_or_bypassrls(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT rolname, rolsuper, rolbypassrls, rolcreaterole, rolcreatedb, "
        "pg_has_role('footnote_app', 'footnote_admin', 'MEMBER') FROM pg_roles "
        "WHERE rolname IN ('footnote_admin', 'footnote_app') ORDER BY rolname",
    )
    assert rows == [
        ("footnote_admin", False, False, False, False, False),
        ("footnote_app", False, False, False, False, False),
    ]


async def test_readyz_role_checks(app_db: Database, admin_db: Database) -> None:
    assert await app_role_problems(app_db) == []
    assert await app_role_problems(admin_db) == [
        "app role owns objects in schema footnote",
        "app role is a member of footnote_admin",
    ]


def test_registered_space_index_matches_spec(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'footnote' AND indexname = %s",
        (f"chunk_vectors_hnsw_s{pg.space_id}",),
    )
    assert rows == [
        (
            f"CREATE INDEX chunk_vectors_hnsw_s{pg.space_id} ON footnote.chunk_vectors USING hnsw "
            f"(((embedding)::vector(8)) vector_cosine_ops) WITH (m='16', ef_construction='64') "
            f"WHERE (space_id = {pg.space_id})",
        )
    ]


def test_keys_and_foreign_keys_start_with_tenant_id(pg: PgInfo) -> None:
    rows = _rows(
        pg.superuser,
        "SELECT c.conrelid::regclass::text, c.conname, c.confrelid::regclass::text, "
        "(SELECT attname FROM pg_attribute WHERE attrelid = c.conrelid AND attnum = c.conkey[1]) "
        "FROM pg_constraint c WHERE c.connamespace = 'footnote'::regnamespace "
        "AND c.contype IN ('u', 'p', 'f')",
    )
    checked = 0
    for table, name, referenced, first in rows:
        if table.removeprefix("footnote.") not in TENANT_TABLES:
            continue
        if referenced == "footnote.embedding_spaces":
            continue  # global table
        expected = "key_hash" if name == "api_keys_key_hash_key" else "tenant_id"
        assert first == expected, (table, name)
        checked += 1
    assert checked > 20
