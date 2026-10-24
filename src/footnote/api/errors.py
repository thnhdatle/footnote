# SPDX-License-Identifier: Apache-2.0
"""HTTP error envelope and the mapping from domain errors to statuses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

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

INTERNAL: Final = "internal"
VALIDATION_FAILED: Final = "validation_failed"

CODE_STATUS: Final[Mapping[str, int]] = {
    "unsupported_media_type": 415,
    "file_too_large": 413,
    "request_too_large": 413,
    "pdf_encrypted": 422,
    "too_many_pages": 422,
    "archive_too_large": 422,
    "parse_failed": 422,
    "embedding_dimension_unsupported": 422,
    "webhook_url_rejected": 422,
    "evidence_threshold_uncalibrated": 409,
    "llm_not_configured": 503,
    "embedding_not_configured": 503,
    "method_not_allowed": 405,
}

CLASS_STATUS: Final[Sequence[tuple[type[DomainError], int]]] = (
    (ValidationError, 422),
    (NotFoundError, 404),
    (ConflictError, 409),
    (QuotaExceededError, 429),
    (UnauthenticatedError, 401),
    (TooLargeError, 413),
    (UnavailableError, 503),
    (UpstreamError, 502),
)

_HTTP_CODES: Final[Mapping[int, str]] = {
    400: "bad_request",
    401: "unauthenticated",
    404: "not_found",
    405: "method_not_allowed",
    413: "request_too_large",
    415: "unsupported_media_type",
    422: VALIDATION_FAILED,
}


def status_for(error: DomainError) -> int:
    if error.code in CODE_STATUS:
        return CODE_STATUS[error.code]
    for cls, status in CLASS_STATUS:
        if isinstance(error, cls):
            return status
    return 500


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def envelope(
    code: str,
    message: str,
    request_id: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        body["details"] = details
    return {"error": body}


def error_response(  # noqa: PLR0913 - envelope fields plus keyword-only extras
    status: int,
    code: str,
    message: str,
    request_id: str,
    *,
    details: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> Response:
    all_headers = dict(headers or {})
    if status == 401:  # noqa: PLR2004 - RFC 6750 challenge on every 401
        all_headers.setdefault("WWW-Authenticate", 'Bearer realm="footnote"')
    return Response(
        orjson.dumps(envelope(code, message, request_id, details)),
        status_code=status,
        media_type="application/json",
        headers=all_headers,
    )


def domain_error_response(error: DomainError, request_id: str) -> Response:
    status = status_for(error)
    if status >= 500 and status not in (502, 503):  # noqa: PLR2004
        return error_response(500, INTERNAL, "internal error", request_id)
    details = None
    if isinstance(error, ValidationError):
        details = [{"field": error.field, "message": error.message}]
    return error_response(status, error.code, error.message, request_id, details=details)


async def _on_domain_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, DomainError)  # noqa: S101 - registered for DomainError only
    return domain_error_response(exc, request_id_of(request))


def _field(loc: Sequence[Any]) -> str:
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "header", "form")]
    return ".".join(parts) or "request"


async def _on_validation_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    details = [{"field": _field(e.get("loc", ())), "message": str(e["msg"])} for e in exc.errors()]
    message = "; ".join(f"{d['field']}: {d['message']}" for d in details[:5])
    return error_response(
        422,
        VALIDATION_FAILED,
        message or "invalid request",
        request_id_of(request),
        details=details,
    )


async def _on_http_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    if isinstance(exc.__cause__, DomainError):
        return domain_error_response(exc.__cause__, request_id_of(request))
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    message = str(exc.detail) if exc.detail else code.replace("_", " ")
    return error_response(
        exc.status_code, code, message, request_id_of(request), headers=exc.headers
    )


def install_error_handlers(app: FastAPI) -> None:
    """Registers the error envelope for domain, validation and routing errors."""
    app.add_exception_handler(DomainError, _on_domain_error)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(StarletteHTTPException, _on_http_error)
