# SPDX-License-Identifier: Apache-2.0
"""Every error code the code base can emit is registered in docs/guides/errors.md,
with the HTTP status the API actually uses."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from footnote.api.errors import CLASS_STATUS, CODE_STATUS, status_for
from footnote.domain.errors import DomainError

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "docs" / "guides" / "errors.md"
_ROW = re.compile(r"^\| `([a-z_]+)` \| (\d{3}) \|", re.MULTILINE)
_ANY_CODE = re.compile(r"^\| `([a-z_]+)` \|", re.MULTILINE)
# `code="…"`, `code = "…"` and module constants holding codes, across src/.
_SOURCE_CODE = re.compile(r"""\bcode\s*=\s*"([a-z_]+)"|^[A-Z_]+ = "([a-z]+(?:_[a-z]+)+)"$""", re.M)
# Constants that look like codes but are not error codes.
_NOT_CODES = {
    "onnx_local",
    "openai_compat",
    "footnote_admin",
    "fts_reconfig",
    "collection_delete",
    "tenant_delete",
    "list_item",
    "ts_rank_cd",
    "insufficient_evidence",
}


def _registry() -> str:
    return REGISTRY.read_text(encoding="utf-8")


def _source_codes() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "src" / "footnote").rglob("*.py"):
        for a, b in _SOURCE_CODE.findall(path.read_text(encoding="utf-8")):
            found.add(a or b)
    return found - _NOT_CODES


def test_registry_lists_every_code_raised_in_source() -> None:
    documented = set(_ANY_CODE.findall(_registry()))
    missing = sorted(_source_codes() - documented)
    assert not missing, f"register in docs/guides/errors.md: {missing}"


def test_registry_lists_every_domain_error_default_code() -> None:
    documented = set(_ANY_CODE.findall(_registry()))
    classes = {cls.code for cls, _ in CLASS_STATUS} | {DomainError.code}
    assert classes <= documented


@pytest.mark.parametrize(("code", "status"), sorted(CODE_STATUS.items()))
def test_registry_status_matches_the_api_for_mapped_codes(code: str, status: int) -> None:
    rows = dict(_ROW.findall(_registry()))
    assert code in rows, code
    assert int(rows[code]) == status


@pytest.mark.parametrize(("cls", "status"), CLASS_STATUS)
def test_registry_status_matches_the_api_for_class_codes(
    cls: type[DomainError], status: int
) -> None:
    rows = dict(_ROW.findall(_registry()))
    assert (
        int(rows[cls.code])
        == status
        == status_for(cls("x") if cls.__init__ is DomainError.__init__ else _make(cls))
    )


def _make(cls: type[DomainError]) -> DomainError:
    from footnote.domain.errors import ValidationError  # noqa: PLC0415

    if issubclass(cls, ValidationError):
        return cls("field", "x")
    return cls("x")
