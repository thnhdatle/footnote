# SPDX-License-Identifier: Apache-2.0
"""Plain text parser."""

from __future__ import annotations

import re

from footnote.adapters.parsers.builder import ElementBuilder
from footnote.domain.documents import Element, ElementKind

_BLANK_LINES = re.compile(r"\n[ \t]*\n\s*")


def parse_txt(text: str) -> list[Element]:
    """Splits text into paragraph elements."""
    builder = ElementBuilder()
    normalised = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    for block in _BLANK_LINES.split(normalised):
        lines = [line.strip() for line in block.split("\n")]
        builder.add(ElementKind.PARAGRAPH, "\n".join(line for line in lines if line))
    return builder.build()
