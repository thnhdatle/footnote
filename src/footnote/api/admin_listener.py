# SPDX-License-Identifier: Apache-2.0
"""Admin listener app."""

from __future__ import annotations

from fastapi import FastAPI

from footnote.api.deps import ApiDeps
from footnote.api.errors import install_error_handlers
from footnote.api.health import health_router
from footnote.api.middleware import RequestContextMiddleware


def create_admin_app(deps: ApiDeps) -> FastAPI:
    """Creates the app served on the admin listener."""
    app = FastAPI(
        title="Footnote admin listener",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        telemetry={"tracing": False},
    )
    app.state.deps = deps
    install_error_handlers(app)
    app.include_router(health_router(include_metrics=True, include_version=False))
    max_body = deps.settings.max_json_bytes
    app.add_middleware(RequestContextMiddleware, deps=deps, body_limit=lambda _scope: max_body)
    return app
