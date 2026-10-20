# SPDX-License-Identifier: Apache-2.0
"""Duration and size parsing."""

from __future__ import annotations

import pytest

from footnote.units import parse_duration, parse_size


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30s", 30.0),
        ("10m", 600.0),
        ("24h", 86400.0),
        ("14d", 14 * 86400.0),
        ("1h30m", 5400.0),
        ("500ms", 0.5),
        (5, 5.0),
        (1.5, 1.5),
    ],
)
def test_parse_duration_accepts(text: object, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "10", "10x", "s", True, None, "1h 30m"])
def test_parse_duration_rejects(text: object) -> None:
    with pytest.raises(ValueError, match="duration"):
        parse_duration(text)


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("64MiB", 64 * 1024**2),
        ("1GB", 10**9),
        ("2 KiB", 2048),
        ("10", 10),
        (7, 7),
        ("1.5KiB", 1536),
    ],
)
def test_parse_size_accepts(text: object, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "MiB", "5XB", True, None, 1.5])
def test_parse_size_rejects(text: object) -> None:
    with pytest.raises(ValueError, match="size"):
        parse_size(text)
