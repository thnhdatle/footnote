# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from footnote.adapters.parsers import media, parse_bytes, sniff

_NAMES = ["code", "long", "nested", "simple", "tables"]
_TYPES = {
    "pdf": media.PDF,
    "docx": media.DOCX,
    "html": media.HTML,
    "md": media.MARKDOWN,
    "txt": media.TEXT,
}


@pytest.mark.parametrize("ext", sorted(_TYPES))
@pytest.mark.parametrize("name", _NAMES)
def test_fixture_matches_expected(corpus: Path, name: str, ext: str) -> None:
    path = corpus / f"{name}.{ext}"
    expected = json.loads((corpus / f"{name}.{ext}.expected.json").read_text(encoding="utf-8"))
    data = path.read_bytes()

    assert sniff(data, path.name) == _TYPES[ext]
    got = [e.to_dict() for e in parse_bytes(_TYPES[ext], data)]

    assert got == expected
