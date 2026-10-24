# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html

from hypothesis import given, settings
from hypothesis import strategies as st

from footnote.adapters.parsers.html_parser import parse_html
from footnote.adapters.parsers.md_parser import parse_markdown
from footnote.adapters.parsers.txt_parser import parse_txt

_PLAIN = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Cc", "Zl", "Zp"), blacklist_characters="\x85"
    ),
    max_size=200,
)
_WORDS = st.lists(st.from_regex(r"[A-Za-zÀ-ÿ0-9]{1,12}", fullmatch=True), min_size=1, max_size=12)
_TAGS = st.sampled_from(
    ["p", "h1", "h3", "li", "ul", "div", "table", "tr", "td", "pre", "nav", "script", "br", "b"]
)
_SOUP = st.lists(
    st.one_of(
        _PLAIN,
        _TAGS.map(lambda t: f"<{t}>"),
        _TAGS.map(lambda t: f"</{t}>"),
        st.just("<!--"),
        st.just("&amp"),
    ),
    max_size=30,
).map("".join)


def _chars(texts: list[str]) -> str:
    return "".join("".join(t.split()) for t in texts)


@settings(max_examples=200, deadline=None)
@given(_PLAIN)
def test_txt_never_raises_and_keeps_every_character(text: str) -> None:
    got = [e.text for e in parse_txt(text)]
    assert _chars(got) == _chars([text.lstrip("﻿")])


@settings(max_examples=200, deadline=None)
@given(_PLAIN | _SOUP)
def test_markdown_never_raises(text: str) -> None:
    parse_markdown(text)


@settings(max_examples=200, deadline=None)
@given(_WORDS, _WORDS)
def test_markdown_keeps_plain_words(heading: list[str], para: list[str]) -> None:
    text = "# " + " ".join(heading) + "\n\n" + " ".join(para) + "\n"
    got = parse_markdown(text)
    assert [e.text for e in got] == [" ".join(heading), " ".join(para)]


@settings(max_examples=200, deadline=None)
@given(_PLAIN | _SOUP)
def test_html_never_raises(text: str) -> None:
    parse_html(text)


@settings(max_examples=200, deadline=None)
@given(st.lists(_PLAIN, min_size=1, max_size=6), st.sampled_from(["p", "li", "h2", "div", "span"]))
def test_html_keeps_text_outside_stripped_tags(texts: list[str], tag: str) -> None:
    body = "".join(f"<{tag}>{html.escape(t)}</{tag}>" for t in texts)
    page = f"<html><body><script>x</script>{body}<nav>gone</nav></body></html>"
    got = [e.text for e in parse_html(page)]
    assert _chars(got) == _chars(texts)
