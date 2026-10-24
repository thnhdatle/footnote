# SPDX-License-Identifier: Apache-2.0
"""Schemas of the job route."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from footnote.domain.jobs import JobRecord


class JobOut(BaseModel):
    """`GET /v1/jobs/{job_id}`."""

    id: UUID
    kind: str
    status: str
    attempts: int
    last_error: str | None
    progress: dict[str, Any] | None
    run_at: datetime
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime

    @classmethod
    def of(cls, j: JobRecord) -> JobOut:
        return cls(
            id=j.id,
            kind=j.kind,
            status=j.status,
            attempts=j.attempts,
            last_error=j.last_error,
            progress=j.progress,
            run_at=j.run_at,
            created_at=j.created_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
            updated_at=j.updated_at,
        )
