# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from uuid import uuid4

from hypothesis import example, given
from hypothesis import strategies as st

from footnote.domain.citations import find_markers, split_sentences, strip_reasoning, verify
from footnote.domain.documents import ChunkId, DocumentId
from footnote.domain.retrieval import Hit
from footnote.domain.webhooks import sign, verify_signature

_ANSWER = st.text(max_size=300) | st.lists(
    st.sampled_from(
        [
            "[1]",
            "[2, 3]",
            "[[1]]",
            "[0]",
            "[999]",
            "[1](u)",
            "```",
            "`",
            ". ",
            "\n\n",
            "- ",
            "e.g. ",
        ]
    )
    | st.text(max_size=12),
    max_size=30,
).map("".join)


def _hit(text: str) -> Hit:
    return Hit(
        chunk_id=ChunkId(uuid4()),
        document_id=DocumentId(uuid4()),
        title="T",
        heading_path=(),
        text=text,
        embedded_text=text,
        page_from=1,
        page_to=1,
    )


@given(_ANSWER, st.lists(st.text(max_size=80), max_size=4))
@example("", [])
@example("See [1] and [2, 3] and [9].", ["Alpha beta."])
@example("[[1]] [1](http://x) `[1]`", ["Alpha."])
@example("```\n[1]", ["Alpha."])
def test_fuzz_verify_keeps_only_valid_markers(answer: str, chunks: list[str]) -> None:
    hits = [_hit(c) for c in chunks]
    got = verify(answer, hits)
    assert all(1 <= n <= len(hits) for n in find_markers(got.text))
    assert got.unsupported == (not got.citations)


@given(_ANSWER)
@example("Dr. Smith said so. It works! Really?")
@example("1. one\n2. two")
def test_fuzz_split_sentences_spans_are_ordered_and_in_range(text: str) -> None:
    spans = split_sentences(text)
    prev = 0
    for s in spans:
        assert prev <= s.start <= s.end <= len(text)
        prev = s.end


@given(st.text(max_size=200))
@example("<think>never ends")
@example("<think>a</think>b")
def test_fuzz_strip_reasoning_never_raises(text: str) -> None:
    assert len(strip_reasoning(text)) <= len(text)


@given(
    st.binary(max_size=64), st.integers(0, 2**40), st.binary(max_size=256), st.text(max_size=100)
)
@example(b"k", 0, b"", "")
@example(b"k", 1, b"{}", "sha256=")
def test_fuzz_verify_signature_rejects_arbitrary_headers(
    secret: bytes, ts: int, body: bytes, header: str
) -> None:
    good = sign(secret, ts, body)
    assert verify_signature(secret, ts, body, good)
    assert verify_signature(secret, ts, body, header) == (header == good)
