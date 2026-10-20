# SPDX-License-Identifier: Apache-2.0
"""Alembic environment: runs as footnote_admin, version table in schema footnote.

The DSN comes from `footnote migrate` (config attribute, resolved from
`FOOTNOTE_ADMIN_DATABASE_URL` or `admin_database.url_file`) or, for plain `alembic`
runs, from the `FOOTNOTE_ADMIN_DATABASE_URL` environment variable.
"""

import os

from alembic import context
from sqlalchemy import create_engine

VERSION_SCHEMA = "footnote"
ADMIN_URL_ATTRIBUTE = "footnote.admin_database_url"


def sqlalchemy_url(url: str) -> str:
    """Select the psycopg 3 driver for a plain postgresql:// DSN."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def admin_url() -> str:
    """The admin DSN: config attribute first, then the environment."""
    url = context.config.attributes.get(ADMIN_URL_ATTRIBUTE) or os.environ.get(
        "FOOTNOTE_ADMIN_DATABASE_URL", ""
    )
    if not url:
        raise SystemExit("FOOTNOTE_ADMIN_DATABASE_URL (or admin_database.url_file) is not set")
    return str(url)


def run_migrations() -> None:
    """Run migrations over the admin DSN."""
    engine = create_engine(sqlalchemy_url(admin_url()))
    try:
        with engine.connect() as conn:
            context.configure(
                connection=conn,
                target_metadata=None,
                version_table_schema=VERSION_SCHEMA,
            )
            with context.begin_transaction():
                context.run_migrations()
            conn.commit()
    finally:
        engine.dispose()


run_migrations()
