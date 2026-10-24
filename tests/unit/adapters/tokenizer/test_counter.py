# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from footnote.adapters.tokenizer import HfTokenCounter
from footnote.domain.errors import UnavailableError


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("empty", "", 0),
        ("one", "a", 1),
        ("several", "a b c d", 4),
        ("unknown words still count", "zzz a", 2),
    ],
)
def test_count(tokenizer_file: Path, name: str, text: str, expected: int) -> None:
    assert HfTokenCounter.from_file(tokenizer_file).count(text) == expected, name


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UnavailableError):
        HfTokenCounter.from_file(tmp_path / "nope.json")
