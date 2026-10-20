# SPDX-License-Identifier: Apache-2.0
"""Shared element assembly: heading paths, ordering and Markdown tables."""

from __future__ import annotations

import re
from collections.abc import Sequence

from footnote.domain.documents import Element, ElementKind

_MONOSPACE = re.compile(r"courier|consolas|mono|menlo|lucida console|source code", re.IGNORECASE)


def is_monospace_font(name: str | None) -> bool:
    """True for font names of common fixed-width families."""
    return bool(name) and _MONOSPACE.search(name or "") is not None


def table_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Markdown table text: header row, `|---|` separator, body rows."""
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + "|".join("---" for _ in rows[0]) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([head, sep, *body])


def squash(text: str) -> str:
    """Collapse every whitespace run to a single space and trim."""
    return " ".join(text.split())


class ElementBuilder:
    """Collects elements, numbering them and tracking the heading path.

    A heading's own `heading_path` includes the heading itself.
    """

    def __init__(self) -> None:
        self._elements: list[Element] = []
        self._stack: list[tuple[int, str]] = []

    def add(
        self,
        kind: ElementKind,
        text: str,
        *,
        level: int | None = None,
        page: int | None = None,
        page_to: int | None = None,
    ) -> None:
        """Append an element; blank text is dropped."""
        if not text.strip():
            return
        if kind is ElementKind.HEADING:
            lvl = level or 1
            while self._stack and self._stack[-1][0] >= lvl:
                self._stack.pop()
            self._stack.append((lvl, text))
        self._elements.append(
            Element(
                kind=kind,
                text=text,
                order=len(self._elements),
                level=level if kind is ElementKind.HEADING else None,
                heading_path=tuple(t for _, t in self._stack),
                page=page,
                page_to=page_to,
            )
        )

    def build(self) -> list[Element]:
        """Return the collected elements."""
        return list(self._elements)
