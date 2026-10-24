# SPDX-License-Identifier: Apache-2.0

import os

from alembic import context
from sqlalchemy import create_engine

VERSION_SCHEMA = "footnote"
ADMIN_URL_ATTRIBUTE = "footnote.admin_database_url"


def sqlalchemy_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def admin_url() -> str:
    url = context.config.attributes.get(ADMIN_URL_ATTRIBUTE) or os.environ.get(
        "FOOTNOTE_ADMIN_DATABASE_URL", ""
    )
    if not url:
        raise SystemExit("FOOTNOTE_ADMIN_DATABASE_URL (or admin_database.url_file) is not set")
    return str(url)


def run_migrations() -> None:
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
