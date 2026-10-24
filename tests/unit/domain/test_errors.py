# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from footnote.domain.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    QuotaExceededError,
    TooLargeError,
    UnauthenticatedError,
    UnavailableError,
    UpstreamError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (NotFoundError, "not_found"),
        (ConflictError, "conflict"),
        (QuotaExceededError, "quota_exceeded"),
        (UnauthenticatedError, "unauthenticated"),
        (UnavailableError, "unavailable"),
        (UpstreamError, "upstream_failed"),
        (TooLargeError, "file_too_large"),
    ],
)
def test_error_default_code(cls: type[DomainError], code: str) -> None:
    err = cls("boom")
    assert err.code == code
    assert err.message == "boom"
    assert str(err) == "boom"
    assert isinstance(err, DomainError)


def test_error_code_override() -> None:
    err = UnavailableError("no llm", code="llm_not_configured")
    assert err.code == "llm_not_configured"
    assert UnavailableError.code == "unavailable"


def test_domain_error_base_code() -> None:
    assert DomainError("x").code == "internal"


def test_validation_error_carries_field_and_prefixes_message() -> None:
    err = ValidationError("retrieval.k", "too big")
    assert err.code == "validation_failed"
    assert err.field == "retrieval.k"
    assert err.message == "retrieval.k: too big"


def test_validation_error_code_override() -> None:
    err = ValidationError("dimensions", "x", code="embedding_dimension_unsupported")
    assert err.code == "embedding_dimension_unsupported"
