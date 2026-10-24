# SPDX-License-Identifier: Apache-2.0
"""PDF parser."""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pdfplumber

from footnote.adapters.parsers.builder import (
    ElementBuilder,
    is_monospace_font,
    squash,
    table_markdown,
)
from footnote.adapters.parsers.limits import ParseLimits, check_pdf
from footnote.domain.documents import Element, ElementKind

_BULLETS = ("•", "◦", "▪", "▫", "●", "○", "■", "□", "‣", "(cid:127)")
_HEADING_MARGIN = 0.5
_NEW_BLOCK_GAP = 1.6
_CODE_BLOCK_GAP = 4.0
_SIZE_EPSILON = 0.05
_SENTENCE_END = (".", "!", "?", ":", "。")
_SPARSE_PAGE_CHARS = 100
_MAX_LEVEL = 6


@dataclass
class _Line:
    page: int
    kind: ElementKind
    text: str
    size: float
    top: float
    bottom: float
    x0: float
    x1: float
    char_width: float
    level: int | None = None


@dataclass
class _Page:
    number: int
    height: float
    lines: list[_Line] = field(default_factory=list)
    tables: list[tuple[float, str]] = field(default_factory=list)
    sizes: Counter[float] = field(default_factory=Counter)


def _inside(char: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _cell(value: object) -> str:
    return squash(str(value or "")).replace("|", "\\|")


def _extract_page(page: Any, number: int) -> _Page:
    out = _Page(number=number, height=float(page.height))
    boxes: list[tuple[float, float, float, float]] = []
    for table in page.find_tables():
        rows = [[_cell(c) for c in row] for row in table.extract() if any(c for c in row)]
        if rows:
            boxes.append(tuple(table.bbox))
            out.tables.append((float(table.bbox[1]), table_markdown(rows)))
    body = page.filter(lambda o: o.get("object_type") != "char" or not _inside(o, boxes))
    for char in body.chars:
        if char["text"].strip():
            out.sizes[round(float(char["size"]), 1)] += 1
    for raw in body.extract_text_lines(layout=False, strip=True, return_chars=True):
        chars = [c for c in raw["chars"] if c["text"].strip()]
        text = raw["text"].strip()
        if not chars or not text:
            continue
        size = Counter(round(float(c["size"]), 1) for c in chars).most_common(1)[0][0]
        mono = sum(is_monospace_font(c["fontname"]) for c in chars) * 2 > len(chars)
        width = (float(raw["x1"]) - float(raw["x0"])) / max(len(text), 1)
        out.lines.append(
            _Line(
                number,
                ElementKind.CODE if mono else ElementKind.PARAGRAPH,
                text,
                size,
                float(raw["top"]),
                float(raw["bottom"]),
                float(raw["x0"]),
                float(raw["x1"]),
                width,
            )
        )
    return out


def _body_size(page: _Page, doc_sizes: Counter[float]) -> float:
    sizes = page.sizes if sum(page.sizes.values()) >= _SPARSE_PAGE_CHARS else doc_sizes
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _classify(pages: list[_Page]) -> None:
    doc_sizes: Counter[float] = Counter()
    for page in pages:
        doc_sizes.update(page.sizes)
    heading_sizes: set[float] = set()
    for page in pages:
        body = _body_size(page, doc_sizes)
        for line in page.lines:
            if line.kind is ElementKind.CODE:
                continue
            if line.size >= body + _HEADING_MARGIN:
                line.kind = ElementKind.HEADING
                heading_sizes.add(line.size)
            elif line.text.startswith(_BULLETS):
                line.kind = ElementKind.LIST_ITEM
                glyph = next(b for b in _BULLETS if line.text.startswith(b))
                line.text = line.text[len(glyph) :].strip()
    ranks = {s: min(i + 1, _MAX_LEVEL) for i, s in enumerate(sorted(heading_sizes, reverse=True))}
    for page in pages:
        for line in page.lines:
            if line.kind is ElementKind.HEADING:
                line.level = ranks[line.size]


@dataclass
class _Block:
    kind: ElementKind
    lines: list[_Line]
    level: int | None = None

    def text(self) -> str:
        if self.kind is ElementKind.CODE:
            return _code_text(self.lines)
        return squash(" ".join(line.text for line in self.lines))


def _code_text(lines: list[_Line]) -> str:
    left = min(line.x0 for line in lines)
    out: list[str] = []
    prev: _Line | None = None
    for line in lines:
        if prev is not None and line.page == prev.page:
            pitch = max(line.size * 1.2, 1.0)
            out.extend("" for _ in range(max(round((line.top - prev.top) / pitch) - 1, 0)))
        indent = round((line.x0 - left) / line.char_width) if line.char_width else 0
        out.append(" " * max(indent, 0) + line.text)
        prev = line
    return "\n".join(out)


def _continues(block: _Block, line: _Line, content_bottom: float) -> bool:
    last = block.lines[-1]
    if block.kind is not line.kind or abs(last.size - line.size) > _SIZE_EPSILON:
        return False
    if block.kind is ElementKind.LIST_ITEM:
        return False
    if line.page != last.page:
        at_bottom = last.bottom >= content_bottom - _NEW_BLOCK_GAP * last.size
        sentence_end = last.text.endswith(_SENTENCE_END) and not line.text[0].islower()
        return block.kind is not ElementKind.HEADING and (
            block.kind is ElementKind.CODE or (at_bottom and not sentence_end)
        )
    gap = _CODE_BLOCK_GAP if block.kind is ElementKind.CODE else _NEW_BLOCK_GAP
    return line.top - last.top <= gap * line.size


def _blocks(pages: list[_Page]) -> list[tuple[int, _Block | str]]:
    content_bottom = max((ln.bottom for p in pages for ln in p.lines), default=0.0)
    out: list[tuple[int, _Block | str]] = []
    current: _Block | None = None
    for page in pages:
        items: list[tuple[float, _Line | str]] = [(ln.top, ln) for ln in page.lines]
        items += [(top, md) for top, md in page.tables]
        for _, item in sorted(items, key=lambda it: it[0]):
            if isinstance(item, str):
                out.append((page.number, item))
                current = None
            elif current is not None and _continues(current, item, content_bottom):
                current.lines.append(item)
            else:
                current = _Block(item.kind, [item])
                out.append((page.number, current))
    return out


def parse_pdf(data: bytes, limits: ParseLimits | None = None) -> list[Element]:
    """Parses PDF bytes into elements."""
    check_pdf(data, limits or ParseLimits())
    pages: list[_Page] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            pages.append(_extract_page(page, number))
            page.close()
    _classify(pages)
    builder = ElementBuilder()
    for page_no, item in _blocks(pages):
        if isinstance(item, str):
            builder.add(ElementKind.TABLE, item, page=page_no, page_to=page_no)
            continue
        first, last = item.lines[0], item.lines[-1]
        builder.add(item.kind, item.text(), level=first.level, page=first.page, page_to=last.page)
    return builder.build()
