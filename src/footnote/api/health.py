# SPDX-License-Identifier: Apache-2.0
"""Health, readiness, metrics and version endpoints."""

from __future__ import annotations

import asyncio
import platform
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response

from footnote.api.deps import ApiDeps, ReadinessCheck, get_deps
from footnote.api.schemas.common import Health, VersionOut

Deps = Annotated[ApiDeps, Depends(get_deps)]
_HEALTH: dict[int | str, dict[str, Any]] = {503: {"model": Health}}


async def _run(check: ReadinessCheck, timeout: float) -> list[str]:
    try:
        async with asyncio.timeout(timeout):
            return list(await check.probe())
    except TimeoutError:
        return [f"timed out after {timeout:g}s"]
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"]


async def readiness(deps: ApiDeps) -> dict[str, list[str]]:
    timeout = deps.settings.readiness_timeout
    results = await asyncio.gather(*(_run(c, timeout) for c in deps.readiness))
    return {c.name: r for c, r in zip(deps.readiness, results, strict=True)}


def health_router(*, include_metrics: bool, include_version: bool) -> APIRouter:
    """Builds the router of the standard endpoints."""
    router = APIRouter(tags=["operations"])

    @router.get("/healthz", response_model=Health, responses=_HEALTH)
    async def healthz(request: Request, response: Response) -> Health:
        """Process alive (503 while draining for shutdown)."""
        if getattr(request.app.state, "draining", False):
            response.status_code = 503
            return Health(status="draining")
        return Health(status="ok")

    @router.get("/readyz", response_model=Health, responses=_HEALTH)
    async def readyz(response: Response, deps: Deps) -> Health:
        """Dependencies reachable and configuration sound; 503 with the problems otherwise."""
        checks = await readiness(deps)
        if any(checks.values()):
            response.status_code = 503
            return Health(status="not_ready", checks=checks)
        return Health(status="ready", checks=checks)

    if include_metrics:

        @router.get(
            "/metrics",
            response_class=Response,
            responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}},
        )
        async def metrics(deps: Deps) -> Response:
            body, content_type = deps.metrics.render()
            return Response(body, media_type=content_type)

    if include_version:

        @router.get("/v1/version", response_model=VersionOut)
        async def version(deps: Deps) -> VersionOut:
            """Build information."""
            v = deps.settings.version
            return VersionOut(
                version=v.version,
                commit=v.commit,
                build_date=v.build_date,
                python_version=platform.python_version(),
            )

    return router
