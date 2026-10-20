# SPDX-License-Identifier: Apache-2.0
"""Media-type dispatch: bytes in, domain elements out (runs inside the sandbox child)."""

from __future__ import annotations

from footnote.adapters.parsers import media
from footnote.adapters.parsers.errors import PARSE_FAILED, UNSUPPORTED_MEDIA_TYPE, ParseError
from footnote.adapters.parsers.limits import ParseLimits, check_docx
from footnote.domain.documents import Element


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(f"not valid UTF-8: {exc}") from exc


def parse_bytes(media_type: str, data: bytes, limits: ParseLimits | None = None) -> list[Element]:
    """Parse `data` of a sniffed `media_type`; raises `ParseError` on any failure."""
    limits = limits or ParseLimits()
    try:
        return _dispatch(media_type, data, limits)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"{type(exc).__name__}: {exc}", code=PARSE_FAILED) from exc


def _dispatch(media_type: str, data: bytes, limits: ParseLimits) -> list[Element]:
    # Imports are lazy so the sandbox child only loads the parser it needs.
    if media_type == media.PDF:
        from footnote.adapters.parsers.pdf_parser import parse_pdf  # noqa: PLC0415

        return parse_pdf(data, limits)
    if media_type == media.DOCX:
        from footnote.adapters.parsers.docx_parser import parse_docx  # noqa: PLC0415

        check_docx(data, limits)
        return parse_docx(data)
    if media_type == media.HTML:
        from footnote.adapters.parsers.html_parser import parse_html  # noqa: PLC0415

        return parse_html(_decode(data))
    if media_type == media.MARKDOWN:
        from footnote.adapters.parsers.md_parser import parse_markdown  # noqa: PLC0415

        return parse_markdown(_decode(data))
    if media_type == media.TEXT:
        from footnote.adapters.parsers.txt_parser import parse_txt  # noqa: PLC0415

        return parse_txt(_decode(data))
    raise ParseError(f"unsupported media type {media_type}", code=UNSUPPORTED_MEDIA_TYPE)
