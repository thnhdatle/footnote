# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from footnote.adapters.parsers import ParseLimits, SandboxedParser, media, parse_bytes
from footnote.domain.documents import ElementKind

ON = ParseLimits(max_images=50)


def _png(width: int, height: int) -> ImageReader:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "teal").save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def _pdf(pages: list[list[tuple[int, int]]], *, text: bool = True) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(600, 800))
    for images in pages:
        if text:
            pdf.setFont("Helvetica-Bold", 20)
            pdf.drawString(50, 750, "Revenue")
            pdf.setFont("Helvetica", 11)
            pdf.drawString(50, 720, "Before the chart.")
            pdf.drawString(50, 100, "After the chart.")
        for n, (width, height) in enumerate(images):
            pdf.drawImage(_png(width, height), 50 + n * 170, 300, width / 2, height / 2)
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def test_image_is_extracted_in_place_with_its_page() -> None:
    parsed = parse_bytes(media.PDF, _pdf([[], [(200, 100)]]), ON)

    [image] = parsed.images
    element = parsed.elements[image.order]
    assert (image.page, image.media_type) == (2, "image/png")
    assert Image.open(io.BytesIO(image.data)).size == (200, 100)
    assert (element.kind, element.page, element.page_to) == (ElementKind.IMAGE, 2, 2)
    assert element.heading_path == ("Revenue",)
    page_two = [e.text for e in parsed.elements if e.page == 2]
    assert page_two == ["Revenue", "Before the chart.", "", "After the chart."]


def test_small_images_are_skipped() -> None:
    parsed = parse_bytes(media.PDF, _pdf([[(32, 300), (300, 300)]]), ON)

    assert len(parsed.images) == 1
    assert Image.open(io.BytesIO(parsed.images[0].data)).size == (300, 300)


def test_images_per_page_are_capped() -> None:
    limits = ParseLimits(max_images=50, max_images_per_page=2)

    parsed = parse_bytes(media.PDF, _pdf([[(100, 100)] * 3]), limits)

    assert len(parsed.images) == 2


def test_images_per_document_are_capped() -> None:
    limits = ParseLimits(max_images=3, max_images_per_page=2)

    parsed = parse_bytes(media.PDF, _pdf([[(100, 100)] * 3] * 2), limits)

    assert [i.page for i in parsed.images] == [1, 1, 2]
    assert sum(e.kind is ElementKind.IMAGE for e in parsed.elements) == 3


def test_no_images_without_an_image_budget() -> None:
    parsed = parse_bytes(media.PDF, _pdf([[(200, 100)]]))

    assert parsed.images == []
    assert all(e.kind is not ElementKind.IMAGE for e in parsed.elements)


async def test_scanned_page_image_crosses_the_sandbox() -> None:
    parsed = await SandboxedParser(ON).parse(media.PDF, _pdf([[(600, 800)]], text=False))

    [image] = parsed.images
    assert image.page == 1
    assert [e.kind for e in parsed.elements] == [ElementKind.IMAGE]
