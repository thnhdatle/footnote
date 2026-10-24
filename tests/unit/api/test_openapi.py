# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import orjson
from fastapi.routing import APIRoute

from footnote.api.app import data_plane_routers
from footnote.api.openapi import main, openapi_document, render

COMMITTED = Path(__file__).resolve().parents[3] / "docs" / "api" / "openapi.json"
REGENERATE = "regenerate with: uv run python -m footnote.api.openapi"


def test_openapi_committed_document_matches_generated() -> None:
    assert COMMITTED.read_bytes() == render(openapi_document()), REGENERATE


def test_openapi_every_route_is_documented_and_vice_versa() -> None:
    routes = {
        (method.lower(), route.path)
        for router in data_plane_routers()
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    spec = orjson.loads(COMMITTED.read_bytes())
    documented = {(m, p) for p, item in spec["paths"].items() for m in item}
    assert routes == documented


def test_openapi_is_3_1_with_error_envelope() -> None:
    spec = openapi_document()
    assert spec["openapi"].startswith("3.1")
    assert "ErrorResponse" in spec["components"]["schemas"]
    schemes = spec["components"]["securitySchemes"]
    assert set(schemes) == {"ApiKey", "AdminKey"}


def test_openapi_main_writes_the_document(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "openapi.json"
    assert main([str(target)]) == 0
    assert target.read_bytes() == render(openapi_document())
