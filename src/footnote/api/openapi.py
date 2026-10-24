# SPDX-License-Identifier: Apache-2.0
"""OpenAPI export of the data plane."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import orjson

from footnote.api.app import build_app

DEFAULT_PATH = Path("docs/api/openapi.json")


def _version() -> str:
    try:
        return version("rayld-footnote")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0.0.0"


def openapi_document() -> dict[str, Any]:
    """Returns the OpenAPI document of the data plane."""
    document: dict[str, Any] = build_app(None, version=_version()).openapi()
    return document


def render(document: dict[str, Any]) -> bytes:
    return orjson.dumps(document, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"


def main(argv: list[str] | None = None) -> int:
    """Writes the OpenAPI document to a file."""
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render(openapi_document()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
