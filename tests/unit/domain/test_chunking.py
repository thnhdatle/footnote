# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.chunking import (
    ChunkDiff,
    chunk_elements,
    content_hash,
    diff_hashes,
    embedded_text,
    normalise_text,
)
from footnote.domain.documents import ChunkDraft, Element, ElementKind
from footnote.domain.settings import ChunkingSettings, ChunkingStrategyName

NBSP = chr(0xA0)


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class PunctCounter:
    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))


H, P, L, T, C = (
    ElementKind.HEADING,
    ElementKind.PARAGRAPH,
    ElementKind.LIST_ITEM,
    ElementKind.TABLE,
    ElementKind.CODE,
)


def _el(
    kind: ElementKind,
    text: str,
    path: tuple[str, ...] = (),
    page: int | None = None,
    page_to: int | None = None,
) -> Element:
    return Element(kind, text, 0, 1 if kind is H else None, path, page, page_to)


def _cfg(
    max_tokens: int,
    overlap: int = 0,
    min_tokens: int = 0,
    strategy: ChunkingStrategyName = ChunkingStrategyName.HEADING,
) -> ChunkingSettings:
    return ChunkingSettings(
        strategy=strategy, max_tokens=max_tokens, overlap_tokens=overlap, min_tokens=min_tokens
    )


def _chunk(
    elements: list[Element],
    cfg: ChunkingSettings,
    counter: WordCounter | PunctCounter | None = None,
) -> list[ChunkDraft]:
    return chunk_elements(elements, "T", cfg, counter or WordCounter())


def _texts(drafts: list[ChunkDraft]) -> list[str]:
    return [d.text for d in drafts]


def test_normalise_text_collapses_whitespace_and_nfc() -> None:
    assert normalise_text("  a \t\n b" + NBSP + " c  ") == "a b c"
    assert normalise_text("é") == "é"


def test_embedded_text_layout() -> None:
    assert embedded_text("T", ("A", "B"), "x y") == "T\nA > B\n\nx y"
    assert embedded_text("T", (), "x") == "T\n\n\nx"


def test_content_hash_is_sha256_of_normalised() -> None:
    assert content_hash(" a  b ") == hashlib.sha256(b"a b").hexdigest()


def test_content_hash_distinguishes_text() -> None:
    assert content_hash("a b") != content_hash("a c")


_WORDS = st.sampled_from(["alpha", "café", "beta.", "gamma,", "delta!", "eps?", "z"])
_WS_RUNS = st.sampled_from([" ", "  ", "\n", "\t ", NBSP, "\r\n"])


@given(st.lists(_WORDS, min_size=1, max_size=12), st.data())
def test_content_hash_invariant_under_whitespace_and_nfd(
    words: list[str], data: st.DataObject
) -> None:
    plain = " ".join(words)
    noisy = "".join(w + data.draw(_WS_RUNS) for w in words)
    assert content_hash(plain) == content_hash(noisy)
    assert content_hash(plain) == content_hash(unicodedata.normalize("NFD", noisy))


def test_chunk_elements_heading_packs_section_exactly() -> None:
    els = [
        _el(H, "Intro"),
        _el(P, "a b c d", ("Intro",)),
        _el(P, "e f g", ("Intro",)),
    ]
    out = _chunk(els, _cfg(6))
    assert _texts(out) == ["a b c d", "e f g"]
    assert [d.ordinal for d in out] == [0, 1]
    assert [d.token_count for d in out] == [4, 3]
    assert out[0].heading_path == ("Intro",)
    assert out[0].embedded_text == "T\nIntro\n\na b c d"
    assert out[0].content_hash == content_hash("T\nIntro\n\na b c d")
    assert not out[0].oversize
    assert out[1].overlap_chars == 0


def test_chunk_elements_heading_joins_elements_within_budget() -> None:
    els = [_el(P, "a b", ("S",)), _el(L, "c d", ("S",))]
    out = _chunk(els, _cfg(10))
    assert _texts(out) == ["a b\n\nc d"]


def test_chunk_elements_heading_sections_split_by_path_and_heading() -> None:
    els = [
        _el(H, "A"),
        _el(P, "a1", ("A",)),
        _el(P, "a2", ("A", "x")),
        _el(H, "A"),
        _el(P, "a3", ("A", "x")),
        _el(H, "Empty"),
    ]
    out = _chunk(els, _cfg(50))
    assert _texts(out) == ["a1", "a2", "a3"]
    assert [d.heading_path for d in out] == [("A",), ("A", "x"), ("A", "x")]


def test_chunk_elements_empty_input_and_blank_elements() -> None:
    assert _chunk([], _cfg(10)) == []
    assert _chunk([_el(H, "H"), _el(P, "  \n ", ("H",))], _cfg(10)) == []


def test_chunk_elements_heading_overlap_exact() -> None:
    els = [_el(P, "a b c d", ("S",)), _el(P, "e f g h", ("S",))]
    out = _chunk(els, _cfg(6, overlap=2))
    assert _texts(out) == ["a b c d", "c d\n\ne f g h"]
    assert [d.overlap_chars for d in out] == [0, 5]
    assert out[1].token_count == 6


def test_chunk_elements_heading_overlap_shrinks_to_fit() -> None:
    els = [_el(P, "a b c d", ("S",)), _el(P, "e f g h i", ("S",))]
    out = _chunk(els, _cfg(6, overlap=2))
    assert _texts(out) == ["a b c d", "d\n\ne f g h i"]
    assert out[1].overlap_chars == 3
    full = _chunk([_el(P, "a b c d", ("S",)), _el(P, "e f g h i j", ("S",))], _cfg(6, overlap=2))
    assert _texts(full)[1] == "e f g h i j"
    assert full[1].overlap_chars == 0


def test_chunk_elements_overlap_not_carried_across_sections() -> None:
    els = [_el(P, "a b c d", ("S",)), _el(P, "e f g h", ("T",))]
    out = _chunk(els, _cfg(6, overlap=2))
    assert _texts(out) == ["a b c d", "e f g h"]


def test_chunk_elements_sentence_split() -> None:
    out = _chunk([_el(P, "one two three. four five six. seven", ("S",))], _cfg(5))
    assert _texts(out) == ["one two three.", "four five six. seven"]


def test_chunk_elements_word_split() -> None:
    out = _chunk([_el(P, "a b c d e f g", ("S",))], _cfg(3))
    assert _texts(out) == ["a b c", "d e f", "g"]
    assert all(not d.oversize for d in out)


def test_chunk_elements_table_split_at_rows_and_oversize_row() -> None:
    out = _chunk([_el(T, "x y\nl1 l2 l3 l4 l5\nz", ("S",))], _cfg(3))
    assert _texts(out) == ["x y", "l1 l2 l3 l4 l5", "z"]
    assert [d.oversize for d in out] == [False, True, False]
    assert out[1].token_count == 5


def test_chunk_elements_code_split_at_lines_only() -> None:
    code = "def f():\n    return 1\nx = 2"
    out = _chunk([_el(C, code, ("S",))], _cfg(4))
    assert _texts(out) == ["def f():\n    return 1", "x = 2"]


def test_chunk_elements_small_table_kept_whole() -> None:
    out = _chunk([_el(T, "a b\n\nc d\n", ("S",))], _cfg(10))
    assert _texts(out) == ["a b\n\nc d"]
    out = _chunk([_el(C, "a\n\n\nb d e", ("S",))], _cfg(3))
    assert _texts(out) == ["a", "b d e"]


def test_chunk_elements_punctuation_counter_exact() -> None:
    out = _chunk([_el(P, "Hi, you. Ok!", ("S",))], _cfg(4), PunctCounter())
    assert _texts(out) == ["Hi, you.", "Ok!"]
    assert [d.token_count for d in out] == [4, 2]


def test_chunk_elements_page_ranges() -> None:
    els = [_el(P, "a b", ("S",), 2, 3), _el(P, "c", ("S",), 5), _el(P, "d", ("S",))]
    out = _chunk(els, _cfg(10))
    assert (out[0].page_from, out[0].page_to) == (2, 5)
    none = _chunk([_el(P, "a", ("S",)), _el(P, "b", ("S",))], _cfg(10))
    assert (none[0].page_from, none[0].page_to) == (None, None)
    mixed = _chunk([_el(P, "a", ("S",)), _el(P, "b", ("S",), 4)], _cfg(10))
    assert (mixed[0].page_from, mixed[0].page_to) == (4, 4)


def test_chunk_elements_identical_passages_share_hash() -> None:
    els = [
        _el(H, "A"),
        _el(P, "same words here", ("A",)),
        _el(H, "A"),
        _el(P, "same   words\nhere", ("A",)),
    ]
    out = _chunk(els, _cfg(50))
    assert len(out) == 2
    assert out[0].content_hash == out[1].content_hash
    assert diff_hashes([], out).new == frozenset({out[0].content_hash})


def test_chunk_elements_min_tokens_leaves_unmergeable_chunk() -> None:
    els = [_el(P, "a b c d e f g h i", ("S",)), _el(P, "x y", ("S",))]
    out = _chunk(els, _cfg(10, min_tokens=3))
    assert _texts(out) == ["a b c d e f g h i", "x y"]


def test_chunk_elements_min_tokens_merges_with_next_dropping_overlap() -> None:
    els = [
        _el(P, "a b c d e f", ("S",), 1),
        _el(P, "g", ("S",), 2),
        _el(P, "h i j k", ("S",), 3),
    ]
    out = _chunk(els, _cfg(6, overlap=3, min_tokens=5))
    assert _texts(out) == ["a b c d e f", "g\n\nh i j k"]
    assert [d.overlap_chars for d in out] == [0, 0]
    assert (out[1].page_from, out[1].page_to) == (2, 3)
    assert out[1].token_count == 5
    assert out[1].content_hash == content_hash("T\nS\n\ng\n\nh i j k")


def test_chunk_elements_min_tokens_merges_into_previous_dropping_its_overlap() -> None:
    els = [_el(P, "a b c d e f", ("S",)), _el(P, "g h i", ("S",)), _el(P, "j", ("S",))]
    out = _chunk(els, _cfg(6, overlap=3, min_tokens=5))
    assert _texts(out) == ["a b c d e f", "g h i\n\nj"]
    assert [d.overlap_chars for d in out] == [0, 0]
    assert out[1].token_count == 4


def test_chunk_elements_min_tokens_does_not_cross_sections() -> None:
    els = [_el(P, "a", ("S",)), _el(P, "b", ("T",))]
    assert _texts(_chunk(els, _cfg(10, min_tokens=5))) == ["a", "b"]


def test_chunk_elements_recursive_ignores_sections() -> None:
    els = [_el(H, "A"), _el(P, "a b c", ("A",), 1), _el(H, "B"), _el(P, "d e", ("B",), 2)]
    rec = _chunk(els, _cfg(6, strategy=ChunkingStrategyName.RECURSIVE))
    assert _texts(rec) == ["a b c\n\nd e"]
    assert rec[0].heading_path == ("A",)
    assert (rec[0].page_from, rec[0].page_to) == (1, 2)
    assert len(_chunk(els, _cfg(6))) == 2


def test_chunk_elements_recursive_overlap_and_min_merge() -> None:
    els = [_el(P, "a b c d", ("A",)), _el(P, "e f g h", ("B",))]
    out = _chunk(els, _cfg(6, overlap=2, strategy=ChunkingStrategyName.RECURSIVE))
    assert _texts(out) == ["a b c d", "c d\n\ne f g h"]
    assert [d.heading_path for d in out] == [("A",), ("B",)]
    assert [d.overlap_chars for d in out] == [0, 5]
    merged = _chunk(
        [_el(P, "a b c d e", ("A",)), _el(P, "f g h i j", ("A",)), _el(P, "k", ("B",))],
        _cfg(7, min_tokens=2, strategy=ChunkingStrategyName.RECURSIVE),
    )
    assert _texts(merged) == ["a b c d e", "f g h i j\n\nk"]


def test_chunk_elements_fixed_windows_with_overlap() -> None:
    cfg = _cfg(4, overlap=1, strategy=ChunkingStrategyName.FIXED)
    out = _chunk([_el(P, "a b c d e f g h i j", ("S",))], cfg)
    assert _texts(out) == ["a b c d", "d e f g", "g h i j"]
    assert [d.overlap_chars for d in out] == [0, 2, 2]


def test_chunk_elements_fixed_no_overlap_and_no_min_merge() -> None:
    cfg = _cfg(3, overlap=0, min_tokens=3, strategy=ChunkingStrategyName.FIXED)
    out = _chunk([_el(P, "a b c d", ("S",), 1), _el(P, "e", ("T",), 2)], cfg)
    assert _texts(out) == ["a b c", "d\n\ne"]
    assert [d.heading_path for d in out] == [("S",), ("S",)]
    assert [(d.page_from, d.page_to) for d in out] == [(1, 1), (1, 2)]


def test_chunk_elements_fixed_table_rows_atomic() -> None:
    cfg = _cfg(4, strategy=ChunkingStrategyName.FIXED)
    out = _chunk([_el(P, "a b", ("S",)), _el(T, "r1 c1\nr2 c2", ("S",))], cfg)
    assert _texts(out) == ["a b\n\nr1 c1", "r2 c2"]


def test_chunk_elements_fixed_oversize_row_and_overlap_shrink() -> None:
    cfg = _cfg(3, overlap=1, strategy=ChunkingStrategyName.FIXED)
    out = _chunk([_el(T, "a b\nx y z w v\nc d", ("S",))], cfg)
    assert _texts(out) == ["a b", "x y z w v", "c d"]
    assert [d.oversize for d in out] == [False, True, False]
    assert [d.overlap_chars for d in out] == [0, 0, 0]


def test_chunk_elements_fixed_overlap_never_blocks_progress() -> None:
    cfg = _cfg(3, overlap=1, strategy=ChunkingStrategyName.FIXED)
    out = _chunk([_el(C, "a b\nc d e", ("S",))], cfg)
    assert _texts(out) == ["a b", "c d e"]


def _draft(h: str) -> ChunkDraft:
    return ChunkDraft(0, "t", (), "e", h, 1, None, None)


def test_diff_hashes_exact() -> None:
    diff = diff_hashes(["a", "z"], [_draft("a"), _draft("b"), _draft("b")])
    assert diff == ChunkDiff(frozenset({"a"}), frozenset({"b"}))
    assert diff_hashes([], []) == ChunkDiff(frozenset(), frozenset())


@given(st.sets(st.sampled_from("abcdef")), st.lists(st.sampled_from("abcdef")))
def test_diff_hashes_partitions_draft_hashes(existing: set[str], hashes: list[str]) -> None:
    diff = diff_hashes(existing, [_draft(h) for h in hashes])
    assert diff.reused | diff.new == set(hashes)
    assert not diff.reused & diff.new
    assert diff.reused <= existing
    assert not diff.new & existing


_PROSE = st.lists(_WORDS, min_size=1, max_size=40).map(" ".join)
_LINES = st.lists(
    st.lists(_WORDS, min_size=1, max_size=8).map(" ".join), min_size=1, max_size=8
).map("\n".join)
_PAGE = st.one_of(st.none(), st.integers(1, 5))


@st.composite
def _documents(draw: st.DrawFn) -> list[Element]:
    elements: list[Element] = []
    stack: list[str] = []
    for order in range(draw(st.integers(0, 10))):
        kind = draw(st.sampled_from([H, P, L, T, C, P]))
        page = draw(_PAGE)
        page_to = None if page is None else draw(st.one_of(st.none(), st.integers(page, 6)))
        if kind is H:
            level = draw(st.integers(1, 3))
            text = draw(st.lists(_WORDS, min_size=1, max_size=3).map(" ".join))
            del stack[level - 1 :]
            stack.append(text)
            elements.append(Element(H, text, order, level, tuple(stack), page, page_to))
        else:
            text = draw(_LINES if kind in (T, C) else _PROSE)
            elements.append(Element(kind, text, order, None, tuple(stack), page, page_to))
    return elements


@st.composite
def _configs(draw: st.DrawFn) -> ChunkingSettings:
    max_tokens = draw(st.integers(4, 30))
    return ChunkingSettings(
        strategy=draw(st.sampled_from(list(ChunkingStrategyName))),
        max_tokens=max_tokens,
        overlap_tokens=draw(st.integers(0, max_tokens // 2)),
        min_tokens=draw(st.integers(0, max_tokens)),
    )


_COUNTERS = st.sampled_from([WordCounter(), PunctCounter()])


@given(_documents(), _configs(), _COUNTERS)
def test_chunk_elements_never_exceeds_budget_unless_oversize(
    elements: list[Element], cfg: ChunkingSettings, counter: WordCounter | PunctCounter
) -> None:
    out = chunk_elements(elements, "T", cfg, counter)
    assert [d.ordinal for d in out] == list(range(len(out)))
    for d in out:
        assert d.token_count == counter.count(d.text)
        assert d.token_count <= cfg.max_tokens or d.oversize
        assert 0 <= d.overlap_chars <= len(d.text)
        assert d.content_hash == content_hash(d.embedded_text)


@given(_documents(), _configs(), _COUNTERS)
def test_chunk_elements_reproduces_body_text(
    elements: list[Element], cfg: ChunkingSettings, counter: WordCounter | PunctCounter
) -> None:
    out = chunk_elements(elements, "T", cfg, counter)
    got = normalise_text(" ".join(d.text[d.overlap_chars :] for d in out))
    want = normalise_text(" ".join(e.text for e in elements if e.kind is not H))
    assert got == want


@given(_documents(), _configs())
def test_chunk_elements_never_splits_rows_or_lines(
    elements: list[Element], cfg: ChunkingSettings
) -> None:
    out = chunk_elements(elements, "T", cfg, WordCounter())
    chunk_lines = {
        line.strip() for d in out for line in d.text[d.overlap_chars :].splitlines() if line.strip()
    }
    for e in elements:
        if e.kind in (T, C):
            for line in e.text.splitlines():
                if line.strip() and cfg.overlap_tokens == 0:
                    assert line.strip() in chunk_lines or line.strip() in " ".join(chunk_lines)
