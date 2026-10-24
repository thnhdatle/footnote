# SPDX-License-Identifier: Apache-2.0
"""Job kinds, requests and records for the queue."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from footnote.domain.documents import JobId
from footnote.domain.tenancy import TenantId


class JobKind(StrEnum):
    INGEST = "ingest"
    REINDEX = "reindex"
    FTS_RECONFIG = "fts_reconfig"
    REEMBED = "reembed"
    GC = "gc"
    WEBHOOK = "webhook"
    COLLECTION_DELETE = "collection_delete"
    TENANT_DELETE = "tenant_delete"


JOB_KINDS = frozenset(k.value for k in JobKind)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NewJob:
    kind: str
    args: Mapping[str, Any] = field(default_factory=dict)
    delay: timedelta = timedelta(0)


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: JobId
    tenant_id: TenantId
    kind: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: JobId
    tenant_id: TenantId
    kind: str
    args: dict[str, Any]
    status: str
    attempts: int
    last_error: str | None
    progress: dict[str, Any] | None
    run_at: datetime
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


class PermanentJobError(Exception): ...
