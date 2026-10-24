# SPDX-License-Identifier: Apache-2.0
"""Edits calibrated thresholds in the model manifest in place."""

from __future__ import annotations

import math
import re

THRESHOLD_DECIMALS = 4
_SCALE = 10_000
_FLOW_KEY = re.compile(r"(calibrated_threshold:\s*)([^,}\s]+)")
_ITEM = re.compile(r"^(\s*)- id:\s*(\S+)\s*(#.*)?$")


class ManifestEditError(ValueError): ...


def stored_threshold(value: float) -> float:
    units = math.floor(value * _SCALE)
    stored = units / _SCALE
    return stored if stored <= value else (units - 1) / _SCALE


def format_threshold(value: float) -> str:
    return f"{value:.{THRESHOLD_DECIMALS}f}"


def set_file_threshold(text: str, sha256: str, value: float) -> str:
    lines = text.splitlines(keepends=True)
    marker = re.compile(rf"sha256:\s*{re.escape(sha256)}\b")
    hits = [i for i, line in enumerate(lines) if marker.search(line)]
    if len(hits) != 1:
        raise ManifestEditError(
            f"expected one manifest file with sha256 {sha256}, found {len(hits)}"
        )
    i = hits[0]
    line = lines[i]
    literal = format_threshold(value)
    if _FLOW_KEY.search(line):
        lines[i] = _FLOW_KEY.sub(lambda m: f"{m.group(1)}{literal}", line, count=1)
    else:
        close = line.rfind("}")
        if close < 0:
            raise ManifestEditError(f"manifest file {sha256} is not a flow mapping")
        lines[i] = f"{line[:close]}, calibrated_threshold: {literal}{line[close:]}"
    return "".join(lines)


def set_model_threshold(text: str, model_id: str, value: float) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    indent = 0
    for i, line in enumerate(lines):
        m = _ITEM.match(line.rstrip("\r\n"))
        if m and m.group(2) == model_id:
            start, indent = i, len(m.group(1))
            break
    if start is None:
        raise ManifestEditError(f"manifest has no model {model_id!r}")
    key = re.compile(rf"^( {{{indent + 2}}}calibrated_threshold:\s*)(\S+)(.*)$", re.DOTALL)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            depth = len(line) - len(line.lstrip(" "))
            if depth <= indent:
                break
        m = key.match(line)
        if m:
            lines[j] = f"{m.group(1)}{format_threshold(value)}{m.group(3)}"
            return "".join(lines)
    raise ManifestEditError(f"manifest model {model_id!r} has no calibrated_threshold key")
