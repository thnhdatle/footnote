# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from footnote.app.ingest import with_descriptions
from footnote.domain.documents import DocumentStatus, Element, ElementKind, PageImage, VersionId
from footnote.domain.errors import UpstreamError

from .fakes import IMAGE_BLOCK
from .rig import Rig, build, paragraph

TEXT = [paragraph("alpha"), IMAGE_BLOCK, paragraph("beta"), IMAGE_BLOCK]


@dataclass
class FakeVision:
    failing: frozenset[bytes] = frozenset()
    prompts: list[str] = field(default_factory=list)

    async def describe(self, image: PageImage, prompt: str) -> str:
        self.prompts.append(prompt)
        if image.data in self.failing:
            raise UpstreamError("vision request timed out")
        return f"A bar chart from {image.data.decode()} on page {image.page}."


async def _elements(rig: Rig) -> list[Element]:
    up = await rig.index_texts(TEXT)
    return await rig.documents.elements(rig.tenant, VersionId(up.version_id))  # type: ignore[arg-type]


def _pages_citing(rig: Rig, text: str) -> list[tuple[int | None, int | None]]:
    return [
        (m.page_from, m.page_to)
        for (t, _), members in rig.db.members.items()
        if t == rig.tenant
        for m in members
        if text in rig.db.chunks[(t, m.chunk_id)].embedded_text
    ]


def _indexed(rig: Rig) -> str:
    chunk_ids = rig.db.current_chunks(rig.tenant, rig.collection.id)
    return " ".join(rig.db.chunks[(rig.tenant, c)].embedded_text for c in chunk_ids)


async def test_images_are_described_stored_and_indexed_with_their_page() -> None:
    vision = FakeVision()
    rig = build(vision=vision)

    elements = await _elements(rig)

    assert [e.kind for e in elements] == [
        ElementKind.PARAGRAPH,
        ElementKind.IMAGE,
        ElementKind.PARAGRAPH,
        ElementKind.IMAGE,
    ]
    assert [e.order for e in elements] == [0, 1, 2, 3]
    image = elements[1]
    assert image.text == "A bar chart from image 1 on page 2."
    assert (image.page, image.page_to) == (2, 2)
    assert "A bar chart from image 1 on page 2." in _indexed(rig)
    [(page_from, page_to)] = _pages_citing(rig, "A bar chart from image 1 on page 2.")
    assert page_from is not None
    assert page_to is not None
    assert page_from <= 2 <= page_to
    assert len(vision.prompts) == 2
    assert "plain text" in vision.prompts[0]
    assert rig.image_errors == [None, None]
    assert rig.stages == ["parse", "describe", "chunk", "embed", "swap"]


async def test_a_vision_error_skips_that_image_and_indexes_the_rest() -> None:
    rig = build(vision=FakeVision(failing=frozenset({b"image 1"})))

    elements = await _elements(rig)

    assert [e.kind for e in elements] == [
        ElementKind.PARAGRAPH,
        ElementKind.PARAGRAPH,
        ElementKind.IMAGE,
    ]
    assert elements[2].text == "A bar chart from image 3 on page 4."
    assert [e.order for e in elements] == [0, 1, 2]
    indexed = _indexed(rig)
    assert paragraph("alpha") in indexed
    assert paragraph("beta") in indexed
    assert "image 1" not in indexed
    [doc] = await rig.documents.list_page(rig.tenant, rig.collection.id, None, 10)
    assert doc.status is DocumentStatus.READY
    failed, described = rig.image_errors
    assert isinstance(failed, UpstreamError)
    assert described is None


async def test_without_vision_images_are_dropped() -> None:
    rig = build()

    elements = await _elements(rig)

    assert [e.text for e in elements] == [paragraph("alpha"), paragraph("beta")]
    assert rig.image_errors == []
    assert "describe" not in rig.stages


def test_with_descriptions_fills_known_images_and_renumbers() -> None:
    elements = [
        Element(ElementKind.IMAGE, "", 0, page=1),
        Element(ElementKind.PARAGRAPH, "text", 1, page=1),
        Element(ElementKind.IMAGE, "", 2, page=2),
    ]

    got = with_descriptions(elements, {2: "A diagram."})

    assert got == [
        Element(ElementKind.PARAGRAPH, "text", 0, page=1),
        Element(ElementKind.IMAGE, "A diagram.", 1, page=2),
    ]
