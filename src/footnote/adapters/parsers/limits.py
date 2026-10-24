# SPDX-License-Identifier: Apache-2.0
"""Parser limits and the checks that enforce them before parsing."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

from footnote.adapters.parsers.errors import (
    ARCHIVE_TOO_LARGE,
    PARSE_FAILED,
    PDF_ENCRYPTED,
    TOO_MANY_PAGES,
    ParseError,
)
from footnote.adapters.parsers.media import DOCX, PDF

_MIB = 1024 * 1024
_CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class ParseLimits:
    max_pages: int = 2000
    max_uncompressed_bytes: int = 200 * _MIB
    max_archive_members: int = 10000
    parse_timeout: float = 300.0


def check_pdf(data: bytes, limits: ParseLimits) -> int:
    from pypdf import PdfReader  # noqa: PLC0415 (heavy import, parse path only)
    from pypdf.errors import PyPdfError  # noqa: PLC0415

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ParseError("PDF is encrypted", code=PDF_ENCRYPTED)
        pages = len(reader.pages)
    except ParseError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, RecursionError) as exc:
        raise ParseError(f"unreadable PDF: {exc}") from exc
    if pages > limits.max_pages:
        raise ParseError(f"PDF has {pages} pages, limit {limits.max_pages}", code=TOO_MANY_PAGES)
    return pages


def check_docx(data: bytes, limits: ParseLimits) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ParseError(f"not a DOCX archive: {exc}") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > limits.max_archive_members:
            raise ParseError(f"archive has {len(infos)} members", code=ARCHIVE_TOO_LARGE)
        if sum(i.file_size for i in infos) > limits.max_uncompressed_bytes:
            raise ParseError("archive declares too much uncompressed data", code=ARCHIVE_TOO_LARGE)
        if "word/document.xml" not in {i.filename for i in infos}:
            raise ParseError("missing word/document.xml", code=PARSE_FAILED)
        total = 0
        for info in infos:
            try:
                with archive.open(info) as member:
                    while chunk := member.read(_CHUNK):
                        total += len(chunk)
                        if total > limits.max_uncompressed_bytes:
                            raise ParseError(
                                "archive expands beyond the limit", code=ARCHIVE_TOO_LARGE
                            )
            except (
                zipfile.BadZipFile,
                OSError,
                EOFError,
                RuntimeError,
                NotImplementedError,
            ) as exc:
                raise ParseError(f"corrupt archive member {info.filename}: {exc}") from exc


def precheck(media_type: str, data: bytes, limits: ParseLimits) -> None:
    """Runs the synchronous upload checks."""
    if media_type == PDF:
        check_pdf(data, limits)
    elif media_type == DOCX:
        check_docx(data, limits)
