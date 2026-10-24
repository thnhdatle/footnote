# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the CLI commands."""

from __future__ import annotations

import os
from collections.abc import Mapping

import typer

from footnote.config import EXIT_CODE_INVALID, ConfigError, Settings, load_settings

EXIT_FAILURE = 1


def fail(message: str, code: int = EXIT_FAILURE) -> typer.Exit:
    typer.echo(f"footnote: {message}", err=True)
    return typer.Exit(code)


def settings_from(ctx: typer.Context, overrides: Mapping[str, str] | None = None) -> Settings:
    root = ctx.find_root()
    path = root.obj.get("config") if isinstance(root.obj, dict) else None
    env = {**os.environ, **(overrides or {})}
    try:
        return load_settings(path, env=env)
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
