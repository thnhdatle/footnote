# SPDX-License-Identifier: Apache-2.0
"""Admin listener (`admin.listen`, default `127.0.0.1:9094`): `/healthz`, `/readyz`
(every readiness check of `ApiDeps`) and `/metrics` (Prometheus).

Bound to loopback by default and kept free of tenant data, so it carries no
authentication (Prometheus scrapes and `footnote healthcheck` call it); the admin API
itself lives on the data plane under `/v1/admin` and requires the admin key.
"""

from __future__ import annotations

from fastapi import FastAPI

from footnote.api.deps import ApiDeps
from footnote.api.errors import install_error_handlers
from footnote.api.health import health_router
from footnote.api.middleware import RequestContextMiddleware


def create_admin_app(deps: ApiDeps) -> FastAPI:
    """Operations app for the admin listener (no OpenAPI, no tenant routes)."""
    app = FastAPI(
        title="Footnote admin listener",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.deps = deps
    install_error_handlers(app)
    app.include_router(health_router(include_metrics=True, include_version=False))
    max_body = deps.settings.max_json_bytes
    app.add_middleware(RequestContextMiddleware, deps=deps, body_limit=lambda _scope: max_body)
    return app
