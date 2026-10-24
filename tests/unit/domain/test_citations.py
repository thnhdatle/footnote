# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from footnote.domain.citations import (
    MARKER_RE,
    Span,
    find_markers,
    split_sentences,
    strip_reasoning,
    verify,
)
from footnote.domain.documents import ChunkId, DocumentId
from footnote.domain.retrieval import Hit


def make_hit(text: str = "Alpha beta. Gamma delta.", title: str = "T", page: int | None = 1) -> Hit:
    return Hit(
        chunk_id=ChunkId(uuid4()),
        document_id=DocumentId(uuid4()),
        title=title,
        heading_path=(),
        text=text,
        embedded_text=text,
        page_from=page,
        page_to=page,
    )


def hits(count: int) -> list[Hit]:
    return [make_hit() for _ in range(count)]


def sentences(text: str) -> list[str]:
    return [text[s.start : s.end] for s in split_sentences(text)]


def test_strip_reasoning_removes_think_block() -> None:
    assert strip_reasoning("<think>hmm</think>\n\nAnswer.") == "Answer."


def test_strip_reasoning_allows_leading_whitespace() -> None:
    assert strip_reasoning("  \n<think>a\nb</think>  Hi") == "Hi"


def test_strip_reasoning_unterminated_returns_empty() -> None:
    assert strip_reasoning("<think>never ends") == ""


def test_strip_reasoning_leaves_later_think_alone() -> None:
    assert strip_reasoning("  Hi <think>x</think>") == "Hi <think>x</think>"


def test_strip_reasoning_only_first_block() -> None:
    assert strip_reasoning("<think>a</think>X<think>b</think>") == "X<think>b</think>"


def test_split_sentences_basic() -> None:
    text = "One. Two! Three? Four"
    assert sentences(text) == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("Hi. Yo") == [Span(0, 3), Span(4, 6)]


def test_split_sentences_empty_and_blank() -> None:
    assert split_sentences("") == []
    assert split_sentences("  \n ") == []


def test_split_sentences_offsets_trimmed() -> None:
    assert split_sentences("  Hi.  Yo  ") == [Span(2, 5), Span(7, 9)]


def test_split_sentences_markers_and_quotes() -> None:
    text = 'He said "yes." Next [1][2]. Last (see [3].) Done.[4] End'
    assert sentences(text) == [
        'He said "yes."',
        "Next [1][2].",
        "Last (see [3].)",
        "Done.[4]",
        "End",
    ]


def test_split_sentences_marker_after_terminator() -> None:
    assert sentences("Yes.[1, 2] No") == ["Yes.[1, 2]", "No"]


def test_split_sentences_no_split_without_whitespace() -> None:
    assert sentences("a.b c") == ["a.b c"]
    assert sentences("Pi is 3.5 or so. Ok") == ["Pi is 3.5 or so.", "Ok"]


@pytest.mark.parametrize(
    "text",
    [
        "Use e.g. apples.",
        "That is, i.e. this one.",
        "Smith et al. found it.",
        "See Fig. 3 now.",
        "Ask Dr. Who today.",
        "Cats vs. dogs here.",
        "Mr. Smith left.",
        "Mrs. Smith left.",
        "Ms. Smith left.",
        "Prof. Smith left.",
        "(e.g. this) works.",
        'Said "Dr. Who" today.',
        "Figs. 3 and 4 show.",
        "Compare cf. this.",
        "About approx. five.",
    ],
)
def test_split_sentences_abbreviations_do_not_split(text: str) -> None:
    assert sentences(text) == [text]


def test_split_sentences_abbreviation_at_text_start() -> None:
    assert sentences("Dr. Who") == ["Dr. Who"]
    assert sentences(". Who") == [".", "Who"]


def test_split_sentences_exclamation_after_abbreviation_splits() -> None:
    assert sentences("Wow Dr! Next") == ["Wow Dr!", "Next"]


def test_split_sentences_blank_lines() -> None:
    assert sentences("First\n\nSecond\n  \nThird") == ["First", "Second", "Third"]


def test_split_sentences_list_items() -> None:
    text = "Intro\n- a\n* b\n+ c\n1. d\n2) e\n  - f"
    assert sentences(text) == ["Intro", "- a", "* b", "+ c", "1. d", "2) e", "- f"]


def test_split_sentences_plain_newline_does_not_split() -> None:
    assert sentences("line one\nline two\n-x") == ["line one\nline two\n-x"]


def test_split_sentences_numbered_item_keeps_number() -> None:
    assert sentences("10. Alpha. Beta") == ["10. Alpha.", "Beta"]
    assert sentences("In 2020. Next") == ["In 2020.", "Next"]


def test_find_markers_basic() -> None:
    assert find_markers("a [1] b [2, 3] c [1]") == [1, 2, 3, 1]


def test_find_markers_ignores_code_and_links() -> None:
    text = "`[1]` and [2](http://x) and\n```\n[3]\n```\nand [4]"
    assert find_markers(text) == [4]


def test_marker_re_rejects_long_numbers_and_empty() -> None:
    assert MARKER_RE.search("[1234] [] [a]") is None
    assert MARKER_RE.search("[12 , 3456]") is None
    assert MARKER_RE.fullmatch("[12 , 34]") is not None


def test_verify_valid_markers_kept() -> None:
    result = verify("A [1] and B [2].", hits(2))
    assert result.text == "A [1] and B [2]."
    assert [c.n for c in result.citations] == [1, 2]
    assert result.unsupported is False
    assert result.invalid_removed == 0


def test_verify_zero_marker_removed() -> None:
    result = verify("A [0] b", hits(2))
    assert result.text == "A b"
    assert result.invalid_removed == 1
    assert result.unsupported is True
    assert result.citations == ()


def test_verify_out_of_range_marker_removed() -> None:
    result = verify("See [99].", hits(8))
    assert result.text == "See."
    assert result.invalid_removed == 1


def test_verify_marker_without_leading_space() -> None:
    assert verify("See[99]. Ok", hits(1)).text == "See. Ok"


def test_verify_marker_at_start() -> None:
    assert verify("[0] Hello", hits(1)).text == " Hello"


def test_verify_consecutive_invalid_markers() -> None:
    result = verify("A [0] [99] b", hits(1))
    assert result.text == "A b"
    assert result.invalid_removed == 2


def test_verify_list_marker_rewritten() -> None:
    result = verify("Claim [1,2].", hits(2))
    assert result.text == "Claim [1][2]."
    assert [c.n for c in result.citations] == [1, 2]
    assert result.invalid_removed == 0


def test_verify_list_with_invalid_number() -> None:
    result = verify("Claim [1, 9].", hits(8))
    assert result.text == "Claim [1]."
    assert result.invalid_removed == 1


def test_verify_list_all_invalid_removed() -> None:
    result = verify("Claim [0, 9] here", hits(8))
    assert result.text == "Claim here"
    assert result.invalid_removed == 2


def test_verify_list_duplicates_collapsed() -> None:
    assert verify("X [1,1,2]", hits(2)).text == "X [1][2]"


def test_verify_adjacent_markers_kept() -> None:
    assert verify("X [1][2] [3].", hits(3)).text == "X [1][2] [3]."


def test_verify_single_marker_kept_verbatim() -> None:
    assert verify("X [01]", hits(1)).text == "X [01]"


def test_verify_nested_markers_collapse() -> None:
    assert verify("A [[1]] b [[[3]]] c", hits(3)).text == "A [1] b [3] c"
    assert verify("A [[1, 2]]", hits(3)).text == "A [1][2]"
    assert verify("A [[99]] b", hits(3)).text == "A b"


def test_verify_markdown_link_untouched() -> None:
    text = "Read [1](https://example.com) and [99](https://x.org)."
    result = verify(text, hits(1))
    assert result.text == text
    assert result.unsupported is True
    assert result.invalid_removed == 0


def test_verify_inline_code_protected() -> None:
    result = verify("Use `a[99]` and ``b [0] ` c`` then [1].", hits(1))
    assert result.text == "Use `a[99]` and ``b [0] ` c`` then [1]."
    assert [c.n for c in result.citations] == [1]


def test_verify_unclosed_backtick_is_literal() -> None:
    assert verify("odd ` tick [99] here", hits(1)).text == "odd ` tick here"


def test_verify_fenced_code_protected() -> None:
    text = "Intro [99]\n```py\nx = a[0]\n[1]\n```\nAfter [77]\n~~~\n[5]\n~~~"
    result = verify(text, hits(2))
    assert result.text == "Intro\n```py\nx = a[0]\n[1]\n```\nAfter\n~~~\n[5]\n~~~"
    assert result.citations == ()
    assert result.invalid_removed == 2


def test_verify_unterminated_fence_protects_rest() -> None:
    result = verify("A [99]\n```\n[98] [1]", hits(1))
    assert result.text == "A\n```\n[98] [1]"


def test_verify_fence_needs_matching_closer() -> None:
    text = "```\n~~~\n[99]\n```\n[98]"
    assert verify(text, hits(1)).text == "```\n~~~\n[99]\n```\n"


def test_verify_nested_collapse_skips_code() -> None:
    assert verify("`[[1]]` [[1]]", hits(1)).text == "`[[1]]` [1]"


def test_verify_removal_creating_marker_reaches_fixed_point() -> None:
    result = verify("[[0]1] x", hits(1))
    assert result.text == "[1] x"
    assert [c.n for c in result.citations] == [1]
    assert result.invalid_removed == 1


def test_verify_no_citation_is_unsupported() -> None:
    result = verify("Plain answer.", hits(3))
    assert result.text == "Plain answer."
    assert result.unsupported is True
    assert result.citations == ()


def test_verify_no_hits_every_marker_invalid() -> None:
    result = verify("A [1] b", [])
    assert result.text == "A b"
    assert result.unsupported is True


def test_verify_question_echo_markers_removed() -> None:
    text = 'You asked "what is [4] and [42]?" The answer is X [3].'
    result = verify(text, hits(8))
    assert result.text == 'You asked "what is [4] and?" The answer is X [3].'
    assert result.invalid_removed == 1
    assert [c.n for c in result.citations] == [4, 3]


@pytest.mark.parametrize("n", range(1, 13))
def test_verify_see_marker_boundary(n: int) -> None:
    result = verify(f"See [{n}].", hits(8))
    if n <= 8:
        assert result.text == f"See [{n}]."
        assert [c.n for c in result.citations] == [n]
        assert result.invalid_removed == 0
    else:
        assert result.text == "See."
        assert result.citations == ()
        assert result.invalid_removed == 1


def test_verify_citation_fields_from_hit() -> None:
    first = make_hit(title="First", page=3)
    second = make_hit(title="Second", page=None)
    result = verify("B [2] then A [1] then B again [2].", [first, second])
    assert [c.n for c in result.citations] == [2, 1]
    cite = result.citations[1]
    assert cite.chunk_id == first.chunk_id
    assert cite.document_id == first.document_id
    assert cite.title == "First"
    assert (cite.page_from, cite.page_to) == (3, 3)
    other = result.citations[0]
    assert other.title == "Second"
    assert (other.page_from, other.page_to) == (None, None)


def test_verify_quote_picks_best_f1_sentence() -> None:
    chunk = make_hit("The cat sat. Dogs bark loudly at night. Birds sing.")
    result = verify("Dogs bark at night [1].", [chunk])
    cite = result.citations[0]
    assert chunk.text[cite.quote_start : cite.quote_end] == "Dogs bark loudly at night."
    assert (cite.quote_start, cite.quote_end) == (13, 39)


def test_verify_quote_uses_sentence_with_first_occurrence() -> None:
    chunk = make_hit("Red apples grow. Blue whales swim.")
    result = verify("Blue whales swim [1]. Red apples grow [1].", [chunk])
    cite = result.citations[0]
    assert chunk.text[cite.quote_start : cite.quote_end] == "Blue whales swim."


def test_verify_quote_tie_picks_earliest() -> None:
    chunk = make_hit("Alpha one. Alpha two. Alpha three.")
    result = verify("Alpha [1].", [chunk])
    cite = result.citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 10)


def test_verify_quote_no_overlap_picks_first_sentence() -> None:
    chunk = make_hit("Zero here. Nothing there.")
    cite = verify("Unrelated words [1].", [chunk]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 10)


def test_verify_quote_ignores_marker_digits() -> None:
    chunk = make_hit("Totally other. Number 1 is here.")
    cite = verify("Claim [1].", [chunk]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 14)


def test_verify_quote_multiset_overlap() -> None:
    chunk = make_hit("go go go. go stop.")
    cite = verify("go stop [1]", [chunk]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (10, 18)


def test_verify_quote_offsets_with_leading_whitespace() -> None:
    chunk = make_hit("   Hello world. Bye now.")
    cite = verify("Bye now [1]", [chunk]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (16, 24)


def test_verify_empty_chunk_gives_zero_span() -> None:
    cite = verify("Claim [1].", [make_hit("")]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 0)
    cite = verify("Claim [1].", [make_hit("  \n ")]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 0)


def test_verify_marker_between_sentences_uses_preceding_sentence() -> None:
    chunk = make_hit("Red apples grow. Blue whales swim.")
    cite = verify("Blue whales swim. [1]", [chunk]).citations[0]
    assert chunk.text[cite.quote_start : cite.quote_end] == "Red apples grow."


def test_verify_marker_before_any_sentence_uses_whole_text() -> None:
    chunk = make_hit("Red apples grow. Blue whales swim.")
    cite = verify("   [1]\n\n", [chunk]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 16)


def test_verify_quote_with_no_tokens_scores_zero() -> None:
    cite = verify("[1]", [make_hit("...")]).citations[0]
    assert (cite.quote_start, cite.quote_end) == (0, 3)


def test_verify_marker_in_code_never_cited() -> None:
    result = verify("`[1]` only", hits(1))
    assert result.unsupported is True


_words = st.sampled_from(["alpha", "beta", "gamma.", "delta,", "Dr.", "e.g.", "3.5", "x"])
_numbers = st.integers(min_value=0, max_value=150)
_markers = st.one_of(
    _numbers.map(lambda n: f"[{n}]"),
    st.lists(_numbers, min_size=1, max_size=3).map(lambda ns: "[" + ", ".join(map(str, ns)) + "]"),
    _numbers.map(lambda n: f"[[{n}]]"),
    _numbers.map(lambda n: f"[{n}](https://example.com)"),
    _numbers.map(lambda n: f"`[{n}]`"),
    _numbers.map(lambda n: f"\n```\n[{n}]\n```\n"),
)
_texts = st.lists(st.one_of(_words, _markers), max_size=14).map(" ".join)
_hit_counts = st.integers(min_value=0, max_value=10)


@settings(max_examples=200, deadline=None)
@given(text=_texts, count=_hit_counts)
def test_verify_property_no_invalid_marker_survives(text: str, count: int) -> None:
    result = verify(text, hits(count))
    assert all(1 <= n <= count for n in find_markers(result.text))
    assert result.unsupported == (not result.citations)
    assert {c.n for c in result.citations} == set(find_markers(result.text))


@settings(max_examples=200, deadline=None)
@given(text=_texts, count=_hit_counts)
def test_verify_property_idempotent(text: str, count: int) -> None:
    hs = hits(count)
    once = verify(text, hs)
    twice = verify(once.text, hs)
    assert twice.text == once.text
    assert twice.citations == once.citations
    assert twice.invalid_removed == 0


@settings(max_examples=100, deadline=None)
@given(
    text=_texts,
    chunk_texts=st.lists(_texts, min_size=1, max_size=4),
)
def test_verify_property_quote_within_chunk(text: str, chunk_texts: list[str]) -> None:
    hs = [make_hit(t) for t in chunk_texts]
    for cite in verify(text, hs).citations:
        body = hs[cite.n - 1].text
        assert 0 <= cite.quote_start <= cite.quote_end <= len(body)


@settings(max_examples=100, deadline=None)
@given(text=st.text(max_size=80))
def test_split_sentences_property_spans_are_trimmed_and_ordered(text: str) -> None:
    previous_end = 0
    for span in split_sentences(text):
        piece = text[span.start : span.end]
        assert piece == piece.strip()
        assert piece
        assert span.start >= previous_end
        previous_end = span.end
