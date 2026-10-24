# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from footnote.adapters.postgres.codec import decode_elements, encode_elements
from footnote.domain.documents import Element, ElementKind


def test_image_element_round_trips_through_storage() -> None:
    elements = [
        Element(ElementKind.HEADING, "Revenue", 0, level=1, heading_path=("Revenue",), page=14),
        Element(ElementKind.IMAGE, "A bar chart.", 1, heading_path=("Revenue",), page=14),
    ]

    assert decode_elements(encode_elements(elements)) == elements
