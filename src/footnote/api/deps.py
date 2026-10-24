# SPDX-License-Identifier: Apache-2.0
"""Dependencies the composition root passes to the API."""

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
    name: str
    probe: ReadinessProbe


@dataclass(frozen=True, slots=True)
class VersionInfo:
    version: str
    commit: str = "unknown"
    build_date: str = "unknown"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    admin_key: str = ""
    version: VersionInfo = field(default_factory=lambda: VersionInfo("0.0.0"))
    access_log: bool = False
    max_json_bytes: int = MIB
    multipart_overhead_bytes: int = MIB
    readiness_timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class ApiDeps:
    """Everything the HTTP layer needs."""

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
    deps: ApiDeps = request.app.state.deps
    return deps
