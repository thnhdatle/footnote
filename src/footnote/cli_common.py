# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the CLI modules: configuration loading and error exits."""

from __future__ import annotations

import os
from collections.abc import Mapping

import typer

from footnote.config import EXIT_CODE_INVALID, ConfigError, Settings, load_settings

EXIT_FAILURE = 1


def fail(message: str, code: int = EXIT_FAILURE) -> typer.Exit:
    """Print `message` to stderr and return the Exit to raise."""
    typer.echo(f"footnote: {message}", err=True)
    return typer.Exit(code)


def settings_from(ctx: typer.Context, overrides: Mapping[str, str] | None = None) -> Settings:
    """Load the configuration (file from the root `--config`); exit 2 with every problem.

    `overrides` are environment-style values from command flags (flags > env > file).
    """
    root = ctx.find_root()
    path = root.obj.get("config") if isinstance(root.obj, dict) else None
    env = {**os.environ, **(overrides or {})}
    try:
        return load_settings(path, env=env)
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
