# SPDX-License-Identifier: Apache-2.0
"""Data-plane application factory."""

from __future__ import annotations

import re

from fastapi import APIRouter, FastAPI
from starlette.types import Scope

from footnote.api.deps import ApiDeps
from footnote.api.errors import install_error_handlers
from footnote.api.health import health_router
from footnote.api.middleware import RequestContextMiddleware
from footnote.api.routers import admin, chunks, collections, documents, jobs, search

TITLE = "Footnote"
DESCRIPTION = (
    "RAG engine on Postgres alone: hybrid retrieval, verifiable citations, tenant "
    "isolation. Errors use the envelope "
    '`{"error": {"code", "message", "request_id"}}` (docs/guides/errors.md).'
)
_UPLOAD_PATH = re.compile(r"^/v1/collections/[^/]+/documents/?$")


def data_plane_routers() -> list[APIRouter]:
    return [
        collections.router,
        documents.router,
        chunks.router,
        search.router,
        jobs.router,
        admin.router,
        health_router(include_metrics=False, include_version=True),
    ]


def build_app(deps: ApiDeps | None, *, version: str = "0.1.0") -> FastAPI:
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=version if deps is None else deps.settings.version.version,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.deps = deps
    install_error_handlers(app)
    for router in data_plane_routers():
        app.include_router(router)
    if deps is not None:
        settings = deps.settings
        upload_limit = deps.documents.max_file_bytes + settings.multipart_overhead_bytes

        def body_limit(scope: Scope) -> int:
            if scope["method"] == "POST" and _UPLOAD_PATH.match(scope["path"]):
                return upload_limit
            return settings.max_json_bytes

        app.add_middleware(RequestContextMiddleware, deps=deps, body_limit=body_limit)
    return app


def create_app(deps: ApiDeps) -> FastAPI:
    """Creates the data-plane app over its dependencies."""
    return build_app(deps)
