# SPDX-License-Identifier: Apache-2.0
"""Citation verifier: sentence splitting, marker cleanup and quote selection."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import TYPE_CHECKING

from footnote.domain.answer import Citation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from footnote.domain.retrieval import Hit

_NUMS = r"\d{1,3}(?:\s*,\s*\d{1,3})*"

MARKER_RE = re.compile(rf"\[({_NUMS})\](?!\()")
"""A `[n]` or `[n, m]` marker that is not markdown link text."""

_COLLAPSE_RE = re.compile(rf"\[{{2,}}({_NUMS})\]{{2,}}")
_NUM_RE = re.compile(r"\d{1,3}")
_WORD_RE = re.compile(r"\w+")
_CODE_RE = re.compile(
    r"^ {0,3}(?P<f>`{3,}|~{3,})[^\n]*(?:\n.*?^ {0,3}(?P=f)[`~]*[ \t]*$|.*\Z)"
    r"|(?<!`)(?P<t>`+)(?!`).+?(?<!`)(?P=t)(?!`)",
    re.MULTILINE | re.DOTALL,
)
_TERMINATOR_RE = re.compile(rf"(?P<term>[.!?]+)(?:[\"'\u201d\u2019)\]]|\[{_NUMS}\])*(?=\s)")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_LIST_ITEM_RE = re.compile(r"\n(?=[ \t]*(?:[-*+]|\d+[.)])[ \t]+)")
_LIST_NUMBER_RE = re.compile(r"[ \t]*\d+")
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "al", "fig", "figs", "dr", "vs", "mr", "mrs", "ms", "prof", "cf", "approx"}
)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    text: str
    citations: tuple[Citation, ...]
    unsupported: bool
    invalid_removed: int


def strip_reasoning(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith(_THINK_OPEN):
        return stripped
    end = stripped.find(_THINK_CLOSE)
    if end == -1:
        return ""
    return stripped[end + len(_THINK_CLOSE) :].lstrip()


def _is_abbreviation(text: str, term_start: int) -> bool:
    tokens = text[:term_start].split()
    if not tokens:
        return False
    return tokens[-1].lstrip("(\"'\u201c\u2018[").lower() in _ABBREVIATIONS


def _is_list_number(text: str, term_start: int) -> bool:
    line_start = text.rfind("\n", 0, term_start) + 1
    return _LIST_NUMBER_RE.fullmatch(text, line_start, term_start) is not None


def split_sentences(text: str) -> list[Span]:
    cuts = {0, len(text)}
    for m in _TERMINATOR_RE.finditer(text):
        if m.group("term") == "." and (
            _is_abbreviation(text, m.start()) or _is_list_number(text, m.start())
        ):
            continue
        cuts.add(m.end())
    cuts.update(m.start() for m in _BLANK_LINE_RE.finditer(text))
    cuts.update(m.start() for m in _LIST_ITEM_RE.finditer(text))
    ordered = sorted(cuts)
    spans: list[Span] = []
    for begin, finish in pairwise(ordered):
        piece = text[begin:finish]
        trimmed = piece.strip()
        if trimmed:
            start = begin + len(piece) - len(piece.lstrip())
            spans.append(Span(start, start + len(trimmed)))
    return spans


def _code_ranges(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _CODE_RE.finditer(text)]


def _in_code(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


def _numbers(group: str) -> list[int]:
    return [int(x) for x in _NUM_RE.findall(group)]


def find_markers(text: str) -> list[int]:
    ranges = _code_ranges(text)
    found: list[int] = []
    for m in MARKER_RE.finditer(text):
        if not _in_code(m.start(), ranges):
            found.extend(_numbers(m.group(1)))
    return found


def _rewrite(m: re.Match[str], n_hits: int) -> tuple[str, int]:
    nums = _numbers(m.group(1))
    valid = [x for x in dict.fromkeys(nums) if 1 <= x <= n_hits]
    invalid = sum(1 for x in nums if not 1 <= x <= n_hits)
    if len(nums) == 1 and valid:
        return m.group(0), 0
    return "".join(f"[{x}]" for x in valid), invalid


def _clean_pass(text: str, n_hits: int) -> tuple[str, int]:
    ranges = _code_ranges(text)
    text = _COLLAPSE_RE.sub(
        lambda m: m.group(0) if _in_code(m.start(), ranges) else f"[{m.group(1)}]", text
    )
    ranges = _code_ranges(text)
    pieces: list[str] = []
    last = 0
    removed = 0
    for m in MARKER_RE.finditer(text):
        if _in_code(m.start(), ranges):
            continue
        repl, bad = _rewrite(m, n_hits)
        prefix = text[last : m.start()]
        if not repl and prefix.endswith(" "):
            prefix = prefix[:-1]
        pieces += [prefix, repl]
        last = m.end()
        removed += bad
    pieces.append(text[last:])
    return "".join(pieces), removed


def _f1(answer: Counter[str], chunk: Counter[str]) -> Fraction:
    total = answer.total() + chunk.total()
    if total == 0:
        return Fraction(0)
    return Fraction(2 * (answer & chunk).total(), total)


def _tokens(text: str) -> Counter[str]:
    return Counter(_WORD_RE.findall(MARKER_RE.sub(" ", text).lower()))


def _best_quote(answer_sentence: str, chunk_text: str) -> tuple[int, int]:
    answer = _tokens(answer_sentence)
    best: tuple[int, int] | None = None
    best_score = Fraction(-1)
    for span in split_sentences(chunk_text):
        score = _f1(answer, _tokens(chunk_text[span.start : span.end]))
        if score > best_score:
            best, best_score = (span.start, span.end), score
    return best if best is not None else (0, 0)


def _sentence_at(text: str, spans: list[Span], pos: int) -> str:
    chosen: Span | None = None
    for span in spans:
        if span.start > pos:
            break
        chosen = span
    return text if chosen is None else text[chosen.start : chosen.end]


def _first_positions(text: str, n_hits: int) -> dict[int, int]:
    ranges = _code_ranges(text)
    first: dict[int, int] = {}
    for m in MARKER_RE.finditer(text):
        if _in_code(m.start(), ranges):
            continue
        for n in _numbers(m.group(1)):
            if 1 <= n <= n_hits:
                first.setdefault(n, m.start())
    return first


def _citations(text: str, hits: Sequence[Hit]) -> tuple[Citation, ...]:
    spans = split_sentences(text)
    out: list[Citation] = []
    for n, pos in _first_positions(text, len(hits)).items():
        hit = hits[n - 1]
        start, end = _best_quote(_sentence_at(text, spans, pos), hit.text)
        out.append(
            Citation(
                n=n,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                title=hit.title,
                page_from=hit.page_from,
                page_to=hit.page_to,
                quote_start=start,
                quote_end=end,
            )
        )
    return tuple(out)


def verify(answer_text: str, hits: Sequence[Hit]) -> VerifiedAnswer:
    """Removes invalid markers and builds one citation per valid number."""
    text = answer_text
    removed = 0
    while True:
        cleaned, bad = _clean_pass(text, len(hits))
        removed += bad
        if cleaned == text:
            break
        text = cleaned
    citations = _citations(text, hits)
    return VerifiedAnswer(
        text=text, citations=citations, unsupported=not citations, invalid_removed=removed
    )
