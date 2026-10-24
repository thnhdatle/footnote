# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from uuid import UUID

from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.documents import (
    ChunkDraft,
    ChunkId,
    DocumentId,
    DocumentRef,
    Element,
    ElementKind,
    Membership,
    VersionId,
)

_elements = st.builds(
    Element,
    kind=st.sampled_from(ElementKind),
    text=st.text(),
    order=st.integers(min_value=0, max_value=10_000),
    level=st.none() | st.integers(min_value=1, max_value=6),
    heading_path=st.lists(st.text(), max_size=4).map(tuple),
    page=st.none() | st.integers(min_value=1, max_value=500),
    page_to=st.none() | st.integers(min_value=1, max_value=500),
)


@given(_elements)
def test_element_dict_round_trip(element: Element) -> None:
    assert Element.from_dict(element.to_dict()) == element


def test_element_to_dict_is_json_ready() -> None:
    element = Element(ElementKind.HEADING, "Intro", 0, level=1, heading_path=("Intro",), page=2)
    assert element.to_dict() == {
        "kind": "heading",
        "text": "Intro",
        "level": 1,
        "heading_path": ["Intro"],
        "page": 2,
        "page_to": None,
        "order": 0,
    }


def test_element_from_dict_minimal_uses_defaults() -> None:
    element = Element.from_dict({"kind": "paragraph", "text": "x", "order": 3})
    assert element == Element(ElementKind.PARAGRAPH, "x", 3)
    assert element.heading_path == ()


def test_element_from_dict_null_heading_path() -> None:
    element = Element.from_dict({"kind": "code", "text": "x", "order": 1, "heading_path": None})
    assert element.heading_path == ()


def test_chunk_draft_defaults() -> None:
    draft = ChunkDraft(0, "t", (), "t", "h", 1, None, None)
    assert draft.oversize is False
    assert draft.overlap_chars == 0


def test_membership_and_document_ref() -> None:
    m = Membership(VersionId(UUID(int=1)), 0, ChunkId(UUID(int=2)), 1, 2)
    assert m.chunk_id == ChunkId(UUID(int=2))
    ref = DocumentRef(DocumentId(UUID(int=3)), "T")
    assert ref.metadata == {}
    assert DocumentRef(DocumentId(UUID(int=3)), "T").metadata is not ref.metadata
