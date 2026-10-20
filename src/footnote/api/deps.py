# SPDX-License-Identifier: Apache-2.0
"""The API's dependency contract: what the composition root hands to `create_app`.

`ApiDeps` holds use cases (app layer) and the few ports the routers read directly; the
API never imports adapters (import-linter). `bootstrap.py` builds one `ApiDeps` per
process and passes it to both `create_app` (data plane) and `create_admin_app`
(admin listener). Readiness checks are plain async callables so the composition root can
wrap any adapter (pools, migrations, role checks, models, LLM `GET /models`).

    deps = ApiDeps(
        collections=CollectionService(...), documents=DocumentService(...),
        admin=AdminService(...), search=SearchService(...), answers=AnswerService(...),
        jobs=PgJobs(app_db), keys=PgKeyResolver(app_db), metrics=Metrics(version=v),
        logger=make_logger(settings.log, component="api"),
        settings=ApiSettings(admin_key=..., version=VersionInfo(v, commit, date)),
        readiness=(ReadinessCheck("database", check_app_pool), ...),
    )
    app = create_app(deps)              # uvicorn server.listen
    admin = create_admin_app(deps)      # uvicorn admin.listen
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from fastapi import Request
from structlog.typing import FilteringBoundLogger

from footnote.app.admin import AdminService
from footnote.app.answer import AnswerService
from footnote.app.collections import CollectionService
from footnote.app.documents import DocumentService
from footnote.app.search import SearchService
from footnote.domain.ports import JobQueue, KeyResolver
from footnote.observability import Metrics

MIB = 1024 * 1024

type ReadinessProbe = Callable[[], Awaitable[list[str]]]


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    """A named dependency check; the probe returns problems (empty list = ready)."""

    name: str
    probe: ReadinessProbe


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """Build information served at `/v1/version`."""

    version: str
    commit: str = "unknown"
    build_date: str = "unknown"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """HTTP-level options.

    `admin_key` empty disables the admin API (every `/v1/admin` request is 401).
    `max_json_bytes` bounds non-upload request bodies; uploads are bounded by
    `DocumentService.max_file_bytes` plus `multipart_overhead_bytes`.
    """

    admin_key: str = ""
    version: VersionInfo = field(default_factory=lambda: VersionInfo("0.0.0"))
    access_log: bool = False
    max_json_bytes: int = MIB
    multipart_overhead_bytes: int = MIB
    readiness_timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class ApiDeps:
    """Everything the HTTP layer needs (built by the composition root)."""

    collections: CollectionService
    documents: DocumentService
    admin: AdminService
    search: SearchService
    answers: AnswerService
    jobs: JobQueue
    keys: KeyResolver
    metrics: Metrics
    logger: FilteringBoundLogger
    settings: ApiSettings = field(default_factory=ApiSettings)
    readiness: Sequence[ReadinessCheck] = ()


async def get_deps(request: Request) -> ApiDeps:
    """FastAPI dependency: the process-wide `ApiDeps` stored on the app."""
    deps: ApiDeps = request.app.state.deps
    return deps
