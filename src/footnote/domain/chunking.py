# SPDX-License-Identifier: Apache-2.0
"""Chunking strategies, text normalisation, content hashing and hash diffing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from footnote.domain.documents import ChunkDraft, Element, ElementKind
from footnote.domain.settings import ChunkingSettings, ChunkingStrategyName

_SEP = "\n\n"
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LINE_KINDS = frozenset({ElementKind.TABLE, ElementKind.CODE})


class TokenCounter(Protocol):
    """Counts model tokens of a text, excluding special tokens."""

    def count(self, text: str) -> int:
        """Return the number of tokens in `text`."""
        ...


def normalise_text(text: str) -> str:
    """NFC-normalise, collapse every whitespace run to one space and strip."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def embedded_text(title: str, heading_path: Sequence[str], text: str) -> str:
    """Text that is embedded and hashed: title, heading path and chunk text."""
    return title + "\n" + " > ".join(heading_path) + _SEP + text


def content_hash(embedded: str) -> str:
    """SHA-256 hex of the normalised embedded text (whitespace and NFC/NFD invariant)."""
    return hashlib.sha256(normalise_text(embedded).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkDiff:
    """Draft hashes split into those the document already has and those that are new."""

    reused: frozenset[str]
    new: frozenset[str]


def diff_hashes(existing: Collection[str], drafts: Sequence[ChunkDraft]) -> ChunkDiff:
    """Partition the distinct draft hashes by presence in `existing`."""
    known = set(existing)
    hashes = {d.content_hash for d in drafts}
    return ChunkDiff(
        reused=frozenset(h for h in hashes if h in known),
        new=frozenset(h for h in hashes if h not in known),
    )


@dataclass(frozen=True, slots=True)
class _Piece:
    """A part of one element that fits the budget (or is an indivisible oversize unit)."""

    text: str
    oversize: bool
    path: tuple[str, ...]
    page_from: int | None
    page_to: int | None


@dataclass(slots=True)
class _Work:
    """A chunk under construction: `overlap` (prefix incl. separator) then `body`."""

    overlap: str
    body: str
    oversize: bool
    path: tuple[str, ...]
    page_from: int | None
    page_to: int | None

    @property
    def text(self) -> str:
        """Full chunk text."""
        return self.overlap + self.body


@dataclass(frozen=True, slots=True)
class _Unit:
    """Atomic unit of the fixed strategy: a word, or a table row / code line."""

    text: str
    sep: str
    path: tuple[str, ...]
    page_from: int | None
    page_to: int | None


def _lo(values: Iterable[int | None]) -> int | None:
    return min((v for v in values if v is not None), default=None)


def _hi(values: Iterable[int | None]) -> int | None:
    return max((v for v in values if v is not None), default=None)


def _greedy(units: Sequence[str], sep: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Join consecutive units while the joined text stays within the budget."""
    groups: list[str] = []
    current: str | None = None
    for unit in units:
        if current is not None and counter.count(current + sep + unit) <= max_tokens:
            current += sep + unit
            continue
        if current is not None:
            groups.append(current)
        current = unit
    if current is not None:
        groups.append(current)
    return groups


def _split_prose(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Split by sentences, then by words for a sentence that alone exceeds the budget."""
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    parts: list[str] = []
    for group in _greedy(sentences, " ", max_tokens, counter):
        if counter.count(group) <= max_tokens:
            parts.append(group)
        else:
            parts.extend(_greedy(group.split(), " ", max_tokens, counter))
    return parts


def _lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def _pieces(element: Element, max_tokens: int, counter: TokenCounter) -> list[_Piece]:
    text = element.text.strip()
    if not text:
        return []
    if counter.count(text) <= max_tokens:
        parts = [text]
    elif element.kind in _LINE_KINDS:
        parts = _greedy(_lines(text), "\n", max_tokens, counter)
    else:
        parts = _split_prose(text, max_tokens, counter)
    page_to = element.page_to or element.page
    return [
        _Piece(p, counter.count(p) > max_tokens, element.heading_path, element.page, page_to)
        for p in parts
    ]


def _tail_words(text: str, overlap_tokens: int, counter: TokenCounter) -> list[str]:
    """Longest run of whole trailing words within `overlap_tokens` tokens."""
    words = text.split()
    taken = 0
    while taken < len(words) and (
        counter.count(" ".join(words[len(words) - taken - 1 :])) <= overlap_tokens
    ):
        taken += 1
    return words[len(words) - taken :]


def _overlap_prefix(
    previous: str, piece: _Piece, settings: ChunkingSettings, counter: TokenCounter
) -> str:
    """Overlap prefix (with separator) for a chunk starting with `piece`, shrunk to fit."""
    if piece.oversize:
        return ""
    words = _tail_words(previous, settings.overlap_tokens, counter)
    while words and counter.count(" ".join(words) + _SEP + piece.text) > settings.max_tokens:
        del words[0]
    return " ".join(words) + _SEP if words else ""


def _pack(
    pieces: Sequence[_Piece], settings: ChunkingSettings, counter: TokenCounter
) -> list[_Work]:
    """Greedily pack pieces in order; chunks after the first get an overlap prefix."""
    out: list[_Work] = []
    current: _Work | None = None
    for piece in pieces:
        if current is not None and not piece.oversize:
            joined = current.text + _SEP + piece.text
            if counter.count(joined) <= settings.max_tokens:
                current.body += _SEP + piece.text
                current.page_from = _lo((current.page_from, piece.page_from))
                current.page_to = _hi((current.page_to, piece.page_to))
                continue
        overlap = ""
        if current is not None:
            out.append(current)
            overlap = _overlap_prefix(current.text, piece, settings, counter)
        current = _Work(
            overlap, piece.text, piece.oversize, piece.path, piece.page_from, piece.page_to
        )
    if current is not None:
        out.append(current)
    return out


def _try_merge(
    first: _Work, second: _Work, settings: ChunkingSettings, counter: TokenCounter
) -> _Work | None:
    """Merge two chunks (keeping the first's overlap if it fits), or None if too big."""
    for overlap in (first.overlap, ""):
        merged = _Work(
            overlap,
            first.body + _SEP + second.body,
            False,
            first.path,
            _lo((first.page_from, second.page_from)),
            _hi((first.page_to, second.page_to)),
        )
        if counter.count(merged.text) <= settings.max_tokens:
            return merged
    return None


def _merge_small(
    chunks: list[_Work], settings: ChunkingSettings, counter: TokenCounter
) -> list[_Work]:
    """Merge chunks below `min_tokens` with the next, else the previous, neighbour."""
    i = 0
    while i < len(chunks):
        if counter.count(chunks[i].text) >= settings.min_tokens:
            i += 1
            continue
        if i + 1 < len(chunks):
            merged = _try_merge(chunks[i], chunks[i + 1], settings, counter)
            if merged is not None:
                chunks[i : i + 2] = [merged]
                continue
        merged = _try_merge(chunks[i - 1], chunks[i], settings, counter) if i > 0 else None
        if merged is None:
            i += 1
        else:
            chunks[i - 1 : i + 1] = [merged]
    return chunks


def _sections(elements: Sequence[Element]) -> list[list[Element]]:
    """Group consecutive body elements by heading path; a heading starts a new section."""
    sections: list[list[Element]] = []
    open_section: list[Element] | None = None
    for element in elements:
        if element.kind is ElementKind.HEADING:
            open_section = None
        elif open_section is not None and open_section[0].heading_path == element.heading_path:
            open_section.append(element)
        else:
            open_section = [element]
            sections.append(open_section)
    return sections


def _units(elements: Sequence[Element]) -> list[_Unit]:
    units: list[_Unit] = []
    for element in elements:
        text = element.text.strip()
        if element.kind in _LINE_KINDS:
            parts, sep = _lines(text), "\n"
        else:
            parts, sep = text.split(), " "
        page_to = element.page_to or element.page
        for j, part in enumerate(parts):
            units.append(
                _Unit(part, _SEP if j == 0 else sep, element.heading_path, element.page, page_to)
            )
    return units


def _window_text(units: Sequence[_Unit], start: int, end: int) -> str:
    return units[start].text + "".join(u.sep + u.text for u in units[start + 1 : end])


def _next_start(
    units: Sequence[_Unit], start: int, end: int, settings: ChunkingSettings, counter: TokenCounter
) -> int:
    """Start of the following window: overlap of at most `overlap_tokens`, always advancing."""
    nxt = end
    while (
        nxt - 1 > start
        and counter.count(_window_text(units, nxt - 1, end)) <= settings.overlap_tokens
    ):
        nxt -= 1
    while nxt < end and counter.count(_window_text(units, nxt, end + 1)) > settings.max_tokens:
        nxt += 1
    return nxt


def _fixed(
    units: Sequence[_Unit], settings: ChunkingSettings, counter: TokenCounter
) -> list[_Work]:
    """Windows of `max_tokens` over the unit stream, overlapping by `overlap_tokens`."""
    out: list[_Work] = []
    start = 0
    previous_end = 0
    while start < len(units):
        end = start
        while (
            end < len(units)
            and counter.count(_window_text(units, start, end + 1)) <= settings.max_tokens
        ):
            end += 1
        end = max(end, start + 1)
        text = _window_text(units, start, end)
        cut = 0
        if start < previous_end:
            cut = len(_window_text(units, start, previous_end)) + len(units[previous_end].sep)
        window = units[start:end]
        out.append(
            _Work(
                text[:cut],
                text[cut:],
                counter.count(text) > settings.max_tokens,
                window[0].path,
                _lo(u.page_from for u in window),
                _hi(u.page_to for u in window),
            )
        )
        if end == len(units):
            break
        previous_end = end
        start = _next_start(units, start, end, settings, counter)
    return out


def chunk_elements(
    elements: Sequence[Element],
    title: str,
    settings: ChunkingSettings,
    counter: TokenCounter,
) -> list[ChunkDraft]:
    """Chunk parsed elements with the configured strategy into ordered drafts."""
    body = [e for e in elements if e.kind is not ElementKind.HEADING]
    limit = settings.max_tokens
    works: list[_Work] = []
    if settings.strategy is ChunkingStrategyName.HEADING:
        for section in _sections(elements):
            pieces = [p for e in section for p in _pieces(e, limit, counter)]
            works.extend(_merge_small(_pack(pieces, settings, counter), settings, counter))
    elif settings.strategy is ChunkingStrategyName.RECURSIVE:
        pieces = [p for e in body for p in _pieces(e, limit, counter)]
        works = _merge_small(_pack(pieces, settings, counter), settings, counter)
    else:
        works = _fixed(_units(body), settings, counter)
    drafts: list[ChunkDraft] = []
    for ordinal, work in enumerate(works):
        embedded = embedded_text(title, work.path, work.text)
        drafts.append(
            ChunkDraft(
                ordinal=ordinal,
                text=work.text,
                heading_path=work.path,
                embedded_text=embedded,
                content_hash=content_hash(embedded),
                token_count=counter.count(work.text),
                page_from=work.page_from,
                page_to=work.page_to,
                oversize=work.oversize,
                overlap_chars=len(work.overlap),
            )
        )
    return drafts
