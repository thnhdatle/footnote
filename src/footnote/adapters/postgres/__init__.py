# SPDX-License-Identifier: Apache-2.0
"""Postgres adapters: pools and tenant transactions, repositories and the SearchStore.

Schema, RLS policies and SECURITY DEFINER functions live in `migrations/versions`.
Repositories take a `Database` (pool + transaction policy): the app pool
(footnote_app, RLS-scoped by `Database.tenant_tx`) or the admin pool
(footnote_admin: tenants, keys, space registration, `tenant_delete`).

Usage::

    pool = await open_pool(PoolSettings(conninfo=dsn))
    store = PgSearchStore(Database(pool))
    async with store.snapshot(tenant_id, collection_id) as snap:
        view = await snap.collection()
"""
