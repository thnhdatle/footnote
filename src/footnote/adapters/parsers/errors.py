# SPDX-License-Identifier: Apache-2.0
"""Parser errors."""

from __future__ import annotations

from footnote.domain.errors import DomainError

PARSE_FAILED = "parse_failed"
PARSE_TIMEOUT = "parse_timeout"
PDF_ENCRYPTED = "pdf_encrypted"
TOO_MANY_PAGES = "too_many_pages"
ARCHIVE_TOO_LARGE = "archive_too_large"
UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"


class ParseError(DomainError):
    """Raised when a document cannot be parsed."""

    code = PARSE_FAILED
