# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "make_fixtures.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fixtures", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures"] = module
    spec.loader.exec_module(module)
    return module


mk = _load()


def _expected(out: Path, name: str) -> list[dict[str, object]]:
    data = json.loads((out / f"{name}.expected.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def test_every_source_yields_all_formats(tmp_path: Path) -> None:
    paths = mk.generate(tmp_path)
    sources = sorted(p.stem for p in mk.SOURCES_DIR.glob("*.md"))
    assert len(sources) >= 4
    for name in sources:
        for ext in mk.EXTENSIONS:
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 0
            assert (tmp_path / f"{name}.{ext}.expected.json").exists()
    assert len(paths) == len(sources) * 10


def test_headings_and_tables_in_expected(tmp_path: Path) -> None:
    mk.generate(tmp_path)
    nested = _expected(tmp_path, "nested.html")
    levels = [e["level"] for e in nested if e["kind"] == "heading"]
    assert levels == [1, 2, 3, 3, 4, 2, 3]
    deep = next(e for e in nested if e["text"] == "Rollback Details")
    assert deep["heading_path"] == ["Operations Guide", "Deployment", "Rollout", "Rollback Details"]
    tables = [e for e in _expected(tmp_path, "tables.docx") if e["kind"] == "table"]
    assert len(tables) == 2
    assert tables[0]["text"].splitlines()[:3] == [
        "| Name | Unit | Price |",
        "|---|---|---|",
        "| Espresso | cup | 2.50 |",
    ]
    assert all(e["kind"] == "paragraph" for e in _expected(tmp_path, "tables.txt"))


def test_pdf_long_doc_spans_pages(tmp_path: Path) -> None:
    mk.generate(tmp_path)
    elements = _expected(tmp_path, "long.pdf")
    assert all(isinstance(e["page"], int) for e in elements)
    assert max(e["page_to"] for e in elements) >= 3
    assert any(e["page_to"] > e["page"] for e in elements)
    assert _expected(tmp_path, "long.md")[0]["page"] is None


def test_expected_deterministic(tmp_path: Path) -> None:
    first = mk.generate(tmp_path / "a")
    mk.generate(tmp_path / "b")
    for path in first:
        if path.name.endswith(".expected.json"):
            assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()
