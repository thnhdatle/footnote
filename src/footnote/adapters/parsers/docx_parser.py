# SPDX-License-Identifier: Apache-2.0
"""DOCX via python-docx: Heading 1..6 styles, list paragraphs, code paragraphs, tables.

python-docx parses XML with entity resolution off, no DTD loading and no network access;
archive limits are enforced by `limits.check_docx` before and while reading.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from footnote.adapters.parsers.builder import (
    ElementBuilder,
    is_monospace_font,
    squash,
    table_markdown,
)
from footnote.domain.documents import Element, ElementKind

_HEADING = re.compile(r"^Heading ([1-6])$")
_CODE_STYLES = {"code", "source code", "preformatted text", "html preformatted"}
_W_P = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
_W_TBL = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl"


def _is_code(par: Paragraph) -> bool:
    style = par.style
    if style is not None and (
        (style.name or "").lower() in _CODE_STYLES or is_monospace_font(style.font.name)
    ):
        return True
    runs = [r for r in par.runs if r.text.strip()]
    return bool(runs) and all(is_monospace_font(r.font.name) for r in runs)


def _is_list(par: Paragraph) -> bool:
    ppr = par._p.pPr
    has_num = ppr is not None and ppr.numPr is not None
    return has_num or (par.style is not None and (par.style.name or "").startswith("List"))


def _blocks(doc: object) -> Iterator[Paragraph | Table]:
    for child in doc.element.body.iterchildren():  # type: ignore[attr-defined]
        if child.tag == _W_P:
            yield Paragraph(child, doc)  # type: ignore[arg-type]
        elif child.tag == _W_TBL:
            yield Table(child, doc)  # type: ignore[arg-type]


def parse_docx(data: bytes) -> list[Element]:
    """Parse DOCX bytes into elements (`page` is always None)."""
    doc = Document(io.BytesIO(data))
    builder = ElementBuilder()
    code: list[str] = []

    def flush_code() -> None:
        builder.add(ElementKind.CODE, "\n".join(code).strip("\n"))
        code.clear()

    for block in _blocks(doc):
        if isinstance(block, Table):
            flush_code()
            rows = [[squash(cell.text) for cell in row.cells] for row in block.rows]
            if rows:
                builder.add(ElementKind.TABLE, table_markdown(rows))
            continue
        if _is_code(block):
            code.append(block.text)
            continue
        flush_code()
        text = squash(block.text)
        match = _HEADING.match(block.style.name or "") if block.style is not None else None
        if match:
            builder.add(ElementKind.HEADING, text, level=int(match.group(1)))
        elif _is_list(block):
            builder.add(ElementKind.LIST_ITEM, text)
        else:
            builder.add(ElementKind.PARAGRAPH, text)
    flush_code()
    return builder.build()
