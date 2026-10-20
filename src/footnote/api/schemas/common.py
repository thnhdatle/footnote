# SPDX-License-Identifier: Apache-2.0
"""Shared models: the error envelope, pages and accepted-job responses."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    """Base of request bodies: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class ErrorDetail(BaseModel):
    """One invalid field."""

    field: str
    message: str


class ErrorBody(BaseModel):
    """`error` member of the envelope."""

    code: str = Field(description="Stable snake_case code (docs/guides/errors.md).")
    message: str
    request_id: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """Error envelope of every non-2xx response."""

    error: ErrorBody


class JobAccepted(BaseModel):
    """A started background job (`GET /v1/jobs/{job_id}`)."""

    job_id: UUID


class Health(BaseModel):
    """`/healthz` and `/readyz` body."""

    status: str
    checks: dict[str, list[str]] | None = None


class VersionOut(BaseModel):
    """`/v1/version` body."""

    version: str
    commit: str
    build_date: str
    python_version: str


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` entries for error statuses."""
    return {s: {"model": ErrorResponse} for s in statuses}


class Page[T](BaseModel):
    """A page of a list endpoint; pass `next_cursor` as `?cursor=` for the next one."""

    data: list[T]
    next_cursor: str | None = None
