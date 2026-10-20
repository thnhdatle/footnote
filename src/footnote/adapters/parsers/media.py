# SPDX-License-Identifier: Apache-2.0
"""Supported media types and content sniffing (never trust the client header)."""

from __future__ import annotations

import io
import zipfile

from footnote.adapters.parsers.errors import UNSUPPORTED_MEDIA_TYPE, ParseError

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
HTML = "text/html"
MARKDOWN = "text/markdown"
TEXT = "text/plain"
SUPPORTED = frozenset({PDF, DOCX, HTML, MARKDOWN, TEXT})

_BOM = b"\xef\xbb\xbf"


def sniff(data: bytes, filename: str = "") -> str:
    """Media type from content; the filename only separates HTML/Markdown/TXT text."""
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"PK"):
        if _is_docx(data):
            return DOCX
        raise ParseError("unsupported ZIP content", code=UNSUPPORTED_MEDIA_TYPE)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError("not UTF-8 text", code=UNSUPPORTED_MEDIA_TYPE) from exc
    if "\x00" in text:
        raise ParseError("binary content", code=UNSUPPORTED_MEDIA_TYPE)
    name = filename.lower()
    head = text.lstrip().lower()
    if head.startswith(("<!doctype html", "<html")) or name.endswith((".html", ".htm")):
        return HTML
    if name.endswith((".md", ".markdown")):
        return MARKDOWN
    return TEXT


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False
