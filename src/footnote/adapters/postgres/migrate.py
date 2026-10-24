# SPDX-License-Identifier: Apache-2.0
"""Alembic migrations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from footnote.domain.errors import UnavailableError

ADMIN_URL_ATTRIBUTE = "footnote.admin_database_url"
_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def find_migrations(candidates: Sequence[Path] | None = None) -> Path | None:
    roots = candidates if candidates is not None else (_SOURCE_ROOT, Path.cwd())
    for root in roots:
        path = root / "migrations"
        if (path / "env.py").is_file() and (path / "versions").is_dir():
            return path
    return None


def alembic_config(admin_url: str, migrations: Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations))
    cfg.set_main_option("path_separator", "os")
    cfg.attributes[ADMIN_URL_ATTRIBUTE] = admin_url
    return cfg


def head_revision(migrations: Path | None) -> str | None:
    if migrations is None:
        return None
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations))
    cfg.set_main_option("path_separator", "os")
    return ScriptDirectory.from_config(cfg).get_current_head()


def upgrade(admin_url: str, migrations: Path | None = None, revision: str = "head") -> None:
    """Runs the alembic upgrade as the admin role."""
    location = migrations or find_migrations()
    if location is None:
        raise UnavailableError("migration scripts not found (expected ./migrations)")
    command.upgrade(alembic_config(admin_url, location), revision)
