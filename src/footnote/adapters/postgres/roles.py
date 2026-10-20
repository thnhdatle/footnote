# SPDX-License-Identifier: Apache-2.0
"""Role checks for `/readyz`: the app DSN must not be able to bypass RLS."""

from __future__ import annotations

from footnote.adapters.postgres.session import Database, fetch_one

ADMIN_ROLE = "footnote_admin"

_CHECK = """
SELECT r.rolsuper AS superuser,
       r.rolbypassrls AS bypassrls,
       EXISTS (SELECT 1 FROM pg_catalog.pg_class c
               WHERE c.relnamespace = 'footnote'::regnamespace AND c.relowner = r.oid)
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p
                  WHERE p.pronamespace = 'footnote'::regnamespace AND p.proowner = r.oid)
       OR EXISTS (SELECT 1 FROM pg_catalog.pg_namespace n
                  WHERE n.nspname = 'footnote' AND n.nspowner = r.oid) AS owns_objects,
       CASE WHEN EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %(admin)s)
            THEN pg_catalog.pg_has_role(r.oid, %(admin)s, 'MEMBER') ELSE false END
         AS admin_member
FROM pg_catalog.pg_roles r
WHERE r.rolname = current_user
"""


async def app_role_problems(app: Database) -> list[str]:
    """Reasons the app role is unsafe (empty when it is fine)."""
    async with app.global_tx() as conn:
        row = await fetch_one(conn, _CHECK, {"admin": ADMIN_ROLE})
    if row is None:
        return ["current role not found"]
    labels = {
        "superuser": "app role is superuser",
        "bypassrls": "app role has BYPASSRLS",
        "owns_objects": "app role owns objects in schema footnote",
        "admin_member": f"app role is a member of {ADMIN_ROLE}",
    }
    return [message for key, message in labels.items() if row[key]]
