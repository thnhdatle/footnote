# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import zipfile

from hypothesis import example, given
from hypothesis import strategies as st

from footnote.adapters.parsers import media
from footnote.adapters.parsers.errors import ParseError
from footnote.adapters.parsers.parse import parse_bytes

_MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


_EMPTY_DOCX = _zip({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>"})


def _only_parse_errors(media_type: str, data: bytes) -> None:
    with contextlib.suppress(ParseError):
        parse_bytes(media_type, data)


@given(st.binary(max_size=2048))
@example(b"")
@example(b"%PDF-")
@example(_MINIMAL_PDF)
@example(_MINIMAL_PDF[:40])
def test_fuzz_pdf_fails_only_with_parse_error(data: bytes) -> None:
    _only_parse_errors(media.PDF, data)


@given(st.binary(max_size=2048))
@example(b"")
@example(b"PK\x03\x04")
@example(_EMPTY_DOCX)
@example(_EMPTY_DOCX[:60])
def test_fuzz_docx_fails_only_with_parse_error(data: bytes) -> None:
    _only_parse_errors(media.DOCX, data)


@given(st.binary(max_size=512), st.sampled_from([media.HTML, media.MARKDOWN, media.TEXT]))
@example(b"\xff\xfe\x00", media.TEXT)
@example(b"<html><body>\xc3\x28</body></html>", media.HTML)
def test_fuzz_text_dispatch_fails_only_with_parse_error(data: bytes, media_type: str) -> None:
    _only_parse_errors(media_type, data)
