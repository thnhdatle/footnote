# SPDX-License-Identifier: Apache-2.0
"""FNT-09: Parser fidelity on generated fixtures."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from acceptance.support import thresholds, write_result
from footnote.adapters.parsers import SandboxedParser, sniff

_GENERATOR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "make_fixtures.py"
_PARAGRAPH_LIKE = {"paragraph", "list_item", "code"}
_MAX_EXAMPLES = 10


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _cells(text: str) -> list[list[str]]:
    """Rows of normalised cells from Markdown table text (separator rows dropped)."""
    rows = [[_norm(c) for c in line.strip().strip("|").split("|")] for line in text.splitlines()]
    return [r for r in rows if not all(c and set(c) <= {"-", ":"} for c in r)]


def _generate(out: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_fixtures_fnt09", _GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures_fnt09"] = module
    spec.loader.exec_module(module)
    module.generate(out)


def _pick(items: list[dict[str, Any]], kinds: set[str]) -> list[dict[str, Any]]:
    return [e for e in items if e["kind"] in kinds]


def _heads(items: list[dict[str, Any]]) -> list[Any]:
    return [(_norm(e["text"]), e["level"]) for e in _pick(items, {"heading"})]


def _tables(items: list[dict[str, Any]]) -> list[Any]:
    return [_cells(e["text"]) for e in _pick(items, {"table"})]


def _compare(name: str, label: str, want: list[Any], got: list[Any]) -> tuple[int, list[str]]:
    """Positional comparison: (matches, mismatch messages) over the expected items."""
    ok = 0
    msgs: list[str] = []
    for i, item in enumerate(want):
        other = got[i] if i < len(got) else None
        if other == item:
            ok += 1
        else:
            msgs.append(f"{name} {label}[{i}] want {item} got {other}")
    return ok, msgs


def _ratio(ok: int, total: int) -> float:
    return ok / total if total else 0.0


def test_acc_fnt09_parser_fidelity(tmp_path: Path) -> None:
    """Parse every generated fixture in the sandbox and compare with the sources' elements."""
    limits = thresholds("FNT-09")
    _generate(tmp_path)
    parser = SandboxedParser()
    mismatches: list[str] = []

    files = sorted(p for p in tmp_path.iterdir() if not p.name.endswith(".expected.json"))
    n_head = n_head_ok = n_tab = n_tab_ok = n_order = n_order_ok = n_pdf = n_pdf_ok = 0
    for path in files:
        data = path.read_bytes()
        got: list[dict[str, Any]] = [
            e.to_dict() for e in asyncio.run(parser.parse(sniff(data, path.name), data))
        ]
        exp: list[dict[str, Any]] = json.loads(
            path.with_name(path.name + ".expected.json").read_text(encoding="utf-8")
        )
        is_txt = path.suffix == ".txt"

        if not is_txt:
            ok, msgs = _compare(path.name, "heading", _heads(exp), _heads(got))
            n_head += len(_heads(exp))
            n_head_ok += ok
            mismatches += msgs
            ok, msgs = _compare(path.name, "table", _tables(exp), _tables(got))
            n_tab += len(_tables(exp))
            n_tab_ok += ok
            mismatches += msgs

        kinds = {"paragraph"} if is_txt else _PARAGRAPH_LIKE
        ep = [_norm(e["text"]) for e in _pick(exp, kinds)]
        gp = [_norm(e["text"]) for e in _pick(got, kinds)]
        n_order += 1
        if ep == gp:
            n_order_ok += 1
        else:
            pairs = enumerate(zip(ep, gp, strict=False))
            first = next((i for i, (a, b) in pairs if a != b), None)
            mismatches.append(
                f"{path.name} paragraph order differs (want {len(ep)}, got {len(gp)}, at {first})"
            )

        if path.suffix == ".pdf":
            pages_want = [(e["page"], e["page_to"]) for e in exp]
            pages_got = [(e.get("page"), e.get("page_to")) for e in got]
            ok, msgs = _compare(path.name, "pages", pages_want, pages_got)
            n_pdf += len(exp)
            n_pdf_ok += ok
            mismatches += msgs

    measured: dict[str, Any] = {
        "heading_match": _ratio(n_head_ok, n_head),
        "table_match": _ratio(n_tab_ok, n_tab),
        "paragraph_order_match": _ratio(n_order_ok, n_order),
        "pdf_page_match": _ratio(n_pdf_ok, n_pdf),
        "fixtures": len(files) // 5,
        "formats": len(files),
        "headings": n_head,
        "tables": n_tab,
        "pdf_elements": n_pdf,
        "mismatches": mismatches[:_MAX_EXAMPLES],
    }
    passed = all(measured[k] >= limits[k] for k in limits)
    write_result("FNT-09", measured, passed=passed, details=f"mismatches={len(mismatches)}")
    assert passed, f"FNT-09 below thresholds {limits}: {measured}"
