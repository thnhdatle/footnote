# SPDX-License-Identifier: Apache-2.0
"""Domain errors: each carries a stable snake_case code (docs/guides/errors.md)."""

from __future__ import annotations


class DomainError(Exception):
    """Base of every Footnote domain error; `code` is part of the public API."""

    code: str = "internal"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ValidationError(DomainError):
    """A request or setting value is out of bounds; `field` names the offending path."""

    code = "validation_failed"

    def __init__(self, field: str, message: str, *, code: str | None = None) -> None:
        super().__init__(f"{field}: {message}", code=code)
        self.field = field


class NotFoundError(DomainError):
    """The resource does not exist for this tenant (other tenants' ids look the same)."""

    code = "not_found"


class ConflictError(DomainError):
    """The request conflicts with the current state (e.g. `space_in_use`)."""

    code = "conflict"


class QuotaExceededError(DomainError):
    """A tenant quota would be exceeded."""

    code = "quota_exceeded"


class UnauthenticatedError(DomainError):
    """The credential is unknown, revoked or expired."""

    code = "unauthenticated"


class UnavailableError(DomainError):
    """A required dependency is not configured or not reachable (e.g. `llm_not_configured`)."""

    code = "unavailable"


class UpstreamError(DomainError):
    """A provider (embedder, LLM, webhook receiver) failed after its retry budget."""

    code = "upstream_failed"


class TooLargeError(DomainError):
    """A request body or uploaded file exceeds its size limit."""

    code = "file_too_large"
