# SPDX-License-Identifier: Apache-2.0
"""UploadChecks: the UploadInspector adapter used by the upload API."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from footnote.adapters.parsers import ParseError, ParseLimits, UploadChecks, media


async def test_inspect_pdf_counts_pages(corpus: Path) -> None:
    data = (corpus / "long.pdf").read_bytes()
    result = await UploadChecks(ParseLimits()).inspect(data, "spoofed.txt")
    assert result.media_type == media.PDF
    assert result.pages is not None
    assert result.pages > 2


async def test_inspect_docx(corpus: Path) -> None:
    data = (corpus / "simple.docx").read_bytes()
    result = await UploadChecks(ParseLimits()).inspect(data, "x.pdf")
    assert (result.media_type, result.pages) == (media.DOCX, None)


@pytest.mark.parametrize(
    ("data", "filename", "expected"),
    [
        (b"# Title\n\nText", "notes.md", media.MARKDOWN),
        (b"<!doctype html><p>x</p>", "page", media.HTML),
        (b"plain words", "a.bin", media.TEXT),
    ],
    ids=["markdown", "html", "text"],
)
async def test_inspect_text_types(data: bytes, filename: str, expected: str) -> None:
    assert (await UploadChecks(ParseLimits()).inspect(data, filename)).media_type == expected


@pytest.mark.parametrize(
    ("name", "limits", "code"),
    [
        ("too_many_pages", ParseLimits(max_pages=2), "too_many_pages"),
        ("binary", ParseLimits(), "unsupported_media_type"),
    ],
)
async def test_inspect_rejects(corpus: Path, name: str, limits: ParseLimits, code: str) -> None:
    data = (corpus / "long.pdf").read_bytes() if name == "too_many_pages" else b"\x00\x01\x02"
    with pytest.raises(ParseError) as err:
        await UploadChecks(limits).inspect(data, "f")
    assert err.value.code == code


def test_check_encrypted_pdf(corpus: Path) -> None:
    writer = PdfWriter()
    writer.append(PdfReader(str(corpus / "simple.pdf")))
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ParseError) as err:
        UploadChecks(ParseLimits()).check(buf.getvalue(), "a.pdf")
    assert err.value.code == "pdf_encrypted"


def test_check_docx_archive_limit(corpus: Path) -> None:
    data = (corpus / "simple.docx").read_bytes()
    with pytest.raises(ParseError) as err:
        UploadChecks(ParseLimits(max_archive_members=1)).check(data, "a.docx")
    assert err.value.code == "archive_too_large"
