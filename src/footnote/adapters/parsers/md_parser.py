# SPDX-License-Identifier: Apache-2.0
"""Markdown (CommonMark + tables) via markdown-it-py; front matter is ignored."""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.token import Token

from footnote.adapters.parsers.builder import ElementBuilder, squash, table_markdown
from footnote.domain.documents import Element, ElementKind

_FRONT_MATTER = re.compile(r"\A﻿?---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def _inline_text(token: Token) -> str:
    parts: list[str] = []
    for child in token.children or []:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        elif child.type == "image":
            parts.append(child.content)
    return squash("".join(parts))


class _Walker:
    def __init__(self) -> None:
        self.builder = ElementBuilder()
        self.list_depth = 0
        self.rows: list[list[str]] | None = None

    def run(self, tokens: list[Token]) -> None:
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                self.builder.add(
                    ElementKind.HEADING, _inline_text(tokens[i + 1]), level=int(tok.tag[1])
                )
            elif tok.type in {"bullet_list_open", "ordered_list_open"}:
                self.list_depth += 1
            elif tok.type in {"bullet_list_close", "ordered_list_close"}:
                self.list_depth -= 1
            elif tok.type in {"fence", "code_block"}:
                self.builder.add(ElementKind.CODE, tok.content.removesuffix("\n"))
            elif tok.type == "table_open":
                self.rows = []
            elif tok.type == "tr_open" and self.rows is not None:
                self.rows.append([])
            elif tok.type == "inline" and self.rows is not None:
                self.rows[-1].append(_inline_text(tok))
            elif tok.type == "table_close":
                self._table()
            elif tok.type == "inline" and tokens[i - 1].type == "paragraph_open":
                kind = ElementKind.LIST_ITEM if self.list_depth else ElementKind.PARAGRAPH
                self.builder.add(kind, _inline_text(tok))
            i += 1

    def _table(self) -> None:
        rows, self.rows = self.rows or [], None
        if rows:
            self.builder.add(ElementKind.TABLE, table_markdown(rows))


def parse_markdown(text: str) -> list[Element]:
    """Parse CommonMark text into elements."""
    body = _FRONT_MATTER.sub("", text, count=1)
    tokens = MarkdownIt("commonmark").enable("table").parse(body)
    walker = _Walker()
    walker.run(tokens)
    return walker.builder.build()
