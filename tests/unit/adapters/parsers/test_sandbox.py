# SPDX-License-Identifier: Apache-2.0
"""Sandbox: timeout kill, crash containment, error code propagation and limits."""

from __future__ import annotations

import os
import time

import pytest

from footnote.adapters.parsers import (
    ParseError,
    ParseLimits,
    SandboxedParser,
    media,
    run_sandboxed,
)
from footnote.domain.documents import Element, ElementKind


def _sleep_forever(_media_type: str, _data: bytes, _limits: ParseLimits) -> list[Element]:
    time.sleep(60)
    return []


def _crash(_media_type: str, _data: bytes, _limits: ParseLimits) -> list[Element]:
    os._exit(137)


def _boom(_media_type: str, _data: bytes, _limits: ParseLimits) -> list[Element]:
    raise RuntimeError("boom")


def _typed_error(_media_type: str, _data: bytes, _limits: ParseLimits) -> list[Element]:
    raise ParseError("too big", code="archive_too_large")


def test_parse_timeout_kills_child() -> None:
    started = time.monotonic()
    with pytest.raises(ParseError) as err:
        run_sandboxed(media.TEXT, b"x", ParseLimits(parse_timeout=1.0), _sleep_forever)
    assert err.value.code == "parse_timeout"
    assert time.monotonic() - started < 30


@pytest.mark.parametrize(
    ("name", "runner", "code"),
    [
        ("crash", _crash, "parse_failed"),
        ("exception", _boom, "parse_failed"),
        ("typed error keeps its code", _typed_error, "archive_too_large"),
    ],
)
def test_child_failures_become_parse_errors(name: str, runner: object, code: str) -> None:
    with pytest.raises(ParseError) as err:
        run_sandboxed(media.TEXT, b"x", ParseLimits(), runner)  # type: ignore[arg-type]
    assert err.value.code == code, name


async def test_sandboxed_parser_round_trips_elements() -> None:
    parser = SandboxedParser()

    got = await parser.parse(media.TEXT, b"first\n\nsecond")

    assert [e.text for e in got] == ["first", "second"]
    assert all(e.kind is ElementKind.PARAGRAPH for e in got)
    assert [e.order for e in got] == [0, 1]


async def test_sandboxed_parser_rejects_unsupported_type() -> None:
    with pytest.raises(ParseError) as err:
        await SandboxedParser().parse("application/zip", b"x")
    assert err.value.code == "unsupported_media_type"
