# SPDX-License-Identifier: Apache-2.0
"""Domain errors with stable codes."""

from __future__ import annotations


class DomainError(Exception):
    """Base of the domain errors."""

    code: str = "internal"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ValidationError(DomainError):
    code = "validation_failed"

    def __init__(self, field: str, message: str, *, code: str | None = None) -> None:
        super().__init__(f"{field}: {message}", code=code)
        self.field = field


class NotFoundError(DomainError):
    code = "not_found"


class ConflictError(DomainError):
    code = "conflict"


class QuotaExceededError(DomainError):
    code = "quota_exceeded"


class UnauthenticatedError(DomainError):
    code = "unauthenticated"


class UnavailableError(DomainError):
    code = "unavailable"


class UpstreamError(DomainError):
    code = "upstream_failed"


class TooLargeError(DomainError):
    code = "file_too_large"
