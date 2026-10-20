# SPDX-License-Identifier: Apache-2.0
"""HTML via selectolax: h1-h6, p, li, table and pre; nav/footer/script/style are dropped."""

from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from footnote.adapters.parsers.builder import ElementBuilder, squash, table_markdown
from footnote.domain.documents import Element, ElementKind

_STRIPPED = ("script", "style", "noscript", "template", "nav", "footer")
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_PARAGRAPHS = {"p", "dt", "dd", "figcaption", "caption"}
_LISTS = {"ul", "ol"}
_STRUCTURE = {"table", "pre", *_LISTS}
_BLOCKS = {
    *_HEADINGS,
    *_PARAGRAPHS,
    *_STRUCTURE,
    "li",
    "div",
    "section",
    "article",
    "main",
    "header",
    "aside",
    "body",
    "html",
    "blockquote",
    "form",
    "figure",
    "dl",
    "details",
    "fieldset",
    "address",
    "hr",
    "tr",
    "td",
    "th",
}


def _text_of(node: Node) -> str:
    parts: list[str] = []
    for child in node.iter(include_text=True):
        tag = child.tag
        if tag == "-text":
            parts.append(child.text(deep=False))
        elif tag == "br" or tag in _BLOCKS:
            parts.append(" " + _text_of(child) + " ")
        elif tag != "-comment":
            parts.append(_text_of(child))
    return "".join(parts)


class _Walker:
    def __init__(self) -> None:
        self.builder = ElementBuilder()

    def children(self, node: Node) -> None:
        """Walk a container, turning loose inline runs into paragraphs."""
        buf: list[str] = []
        for child in node.iter(include_text=True):
            tag = child.tag
            if tag == "-text":
                buf.append(child.text(deep=False))
            elif tag == "-comment":
                continue
            elif tag in _BLOCKS:
                self._flush(buf, ElementKind.PARAGRAPH)
                self.block(child)
            else:
                buf.append(_text_of(child))
        self._flush(buf, ElementKind.PARAGRAPH)

    def _flush(self, buf: list[str], kind: ElementKind) -> None:
        self.builder.add(kind, squash("".join(buf)))
        buf.clear()

    def block(self, node: Node) -> None:
        """Handle one block-level element."""
        tag = node.tag
        if tag in _HEADINGS:
            self.builder.add(ElementKind.HEADING, squash(_text_of(node)), level=_HEADINGS[tag])
        elif tag in _PARAGRAPHS:
            self.builder.add(ElementKind.PARAGRAPH, squash(_text_of(node)))
        elif tag == "pre":
            self.builder.add(ElementKind.CODE, node.text(deep=True, strip=False).strip("\n"))
        elif tag == "table":
            self._table(node)
        elif tag == "li":
            self._list_item(node)
        else:
            self.children(node)

    def _list_item(self, node: Node) -> None:
        buf: list[str] = []
        for child in node.iter(include_text=True):
            tag = child.tag
            if tag == "-text":
                buf.append(child.text(deep=False))
            elif tag in _STRUCTURE:
                self._flush(buf, ElementKind.LIST_ITEM)
                self.block(child)
            elif tag != "-comment":
                buf.append(" " + _text_of(child) + " ")
        self._flush(buf, ElementKind.LIST_ITEM)

    def _table(self, node: Node) -> None:
        rows = []
        for tr in node.css("tr"):
            cells = [
                squash(_text_of(c)).replace("|", "\\|") for c in tr.iter() if c.tag in {"th", "td"}
            ]
            if cells:
                rows.append(cells)
        if rows:
            self.builder.add(ElementKind.TABLE, table_markdown(rows))


def parse_html(text: str) -> list[Element]:
    """Parse an HTML document into elements."""
    tree = HTMLParser(text)
    for node in tree.css(",".join(_STRIPPED)):
        node.decompose()
    walker = _Walker()
    if tree.body is not None:
        walker.children(tree.body)
    return walker.builder.build()
