# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from footnote.adapters.parsers import ParseError, ParseLimits, media, parse_bytes, precheck, sniff

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx_with_xml(base: bytes, document_xml: str) -> bytes:
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(base)) as src,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "word/document.xml":
                payload = document_xml.encode()
            dst.writestr(info.filename, payload)
    return out.getvalue()


def _body(text: str) -> str:
    return f"<w:document {_W}><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"


@pytest.fixture
def base_docx(corpus: Path) -> bytes:
    return (corpus / "simple.docx").read_bytes()


def test_billion_laughs_docx_is_not_expanded(base_docx: bytes) -> None:
    entities = ['<!ENTITY lol0 "lol">']
    for i in range(1, 10):
        refs = f"&lol{i - 1};" * 10
        entities.append(f'<!ENTITY lol{i} "{refs}">')
    xml = f"<?xml version='1.0'?><!DOCTYPE d [{''.join(entities)}]>" + _body("&lol9;")
    data = _docx_with_xml(base_docx, xml)

    with contextlib.suppress(ParseError):
        elements = parse_bytes(media.DOCX, data)
        assert sum(len(e.text) for e in elements) < 1000


def test_external_entity_docx_reads_no_file(base_docx: bytes) -> None:
    xml = "<?xml version='1.0'?><!DOCTYPE d [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>" + _body(
        "&xxe;"
    )
    data = _docx_with_xml(base_docx, xml)

    elements = parse_bytes(media.DOCX, data)

    assert "root:" not in " ".join(e.text for e in elements)


@pytest.mark.parametrize(
    ("name", "limits"),
    [
        ("member count", ParseLimits(max_archive_members=3)),
        ("uncompressed size", ParseLimits(max_uncompressed_bytes=1000)),
    ],
)
def test_zip_bomb_limits(base_docx: bytes, name: str, limits: ParseLimits) -> None:
    with pytest.raises(ParseError) as err:
        precheck(media.DOCX, base_docx, limits)
    assert err.value.code == "archive_too_large", name


def test_deflate_bomb_is_stopped_while_reading(base_docx: bytes) -> None:
    big = "<w:t>" + "a" * 2_000_000 + "</w:t>"
    data = _docx_with_xml(base_docx, _body("x").replace("<w:t>x</w:t>", big))
    assert len(data) < 100_000

    with pytest.raises(ParseError) as err:
        parse_bytes(media.DOCX, data, ParseLimits(max_uncompressed_bytes=1_000_000))

    assert err.value.code == "archive_too_large"


def test_corrupt_docx_is_parse_failed() -> None:
    with pytest.raises(ParseError) as err:
        parse_bytes(media.DOCX, b"PK\x03\x04 not really a zip")
    assert err.value.code == "parse_failed"


def test_encrypted_pdf(corpus: Path) -> None:
    writer = PdfWriter()
    writer.append(PdfReader(str(corpus / "simple.pdf")))
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)

    for call in (precheck, lambda m, d, _l: parse_bytes(m, d)):
        with pytest.raises(ParseError) as err:
            call(media.PDF, buf.getvalue(), ParseLimits())
        assert err.value.code == "pdf_encrypted"


def test_too_many_pages(corpus: Path) -> None:
    data = (corpus / "long.pdf").read_bytes()

    with pytest.raises(ParseError) as err:
        precheck(media.PDF, data, ParseLimits(max_pages=2))

    assert err.value.code == "too_many_pages"


def test_garbage_pdf_is_parse_failed() -> None:
    with pytest.raises(ParseError) as err:
        parse_bytes(media.PDF, b"%PDF-1.7 garbage")
    assert err.value.code == "parse_failed"


@pytest.mark.parametrize(
    ("name", "data", "filename", "expected"),
    [
        ("pdf magic wins over name", b"%PDF-1.4\n", "x.txt", media.PDF),
        ("html doctype", b"  <!DOCTYPE HTML><p>x", "x.txt", media.HTML),
        ("html by extension", b"<p>x</p>", "x.htm", media.HTML),
        ("markdown by extension", b"# t", "x.markdown", media.MARKDOWN),
        ("plain default", b"hello", "x.md.exe", media.TEXT),
        ("bom allowed", b"\xef\xbb\xbfhello", "", media.TEXT),
    ],
)
def test_sniff(name: str, data: bytes, filename: str, expected: str) -> None:
    assert sniff(data, filename) == expected, name


def test_sniff_detects_docx_not_other_zip(base_docx: bytes) -> None:
    other = io.BytesIO()
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("a.txt", "x")

    assert sniff(base_docx, "x.txt") == media.DOCX
    for data in (other.getvalue(), b"\xff\xfe\x00binary", b"abc\x00def"):
        with pytest.raises(ParseError) as err:
            sniff(data)
        assert err.value.code == "unsupported_media_type"
