# SPDX-License-Identifier: Apache-2.0
"""Job route."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from footnote.api.auth import Tenant
from footnote.api.deps import ApiDeps, get_deps
from footnote.api.schemas.common import error_responses
from footnote.api.schemas.jobs import JobOut
from footnote.domain.documents import JobId

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut, responses=error_responses(401, 404))
async def get_job(
    job_id: UUID, principal: Tenant, deps: Annotated[ApiDeps, Depends(get_deps)]
) -> JobOut:
    """Kind, status, attempts, last error, progress and timestamps of a job."""
    return JobOut.of(await deps.jobs.get(principal.tenant, JobId(job_id)))
