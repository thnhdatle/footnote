# SPDX-License-Identifier: Apache-2.0
"""Duration and size parsing shared by configuration.

Durations are Go-style strings with a `d` suffix (`"30s"`, `"1h30m"`, `"14d"`) and are
returned as seconds. Sizes are bytes with an optional suffix (`"64MiB"`, `"1.5GB"`).
"""

from __future__ import annotations

import re

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)")
_DURATION_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)")
_SIZE_FACTOR = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}


def parse_duration(value: object) -> float:
    """Seconds for a duration string; plain numbers are taken as seconds."""
    if isinstance(value, bool):
        raise ValueError("not a duration")
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("not a duration")
    text = value.strip()
    total = 0.0
    pos = 0
    while pos < len(text):
        match = _DURATION_PART.match(text, pos)
        if match is None:
            raise ValueError(f"invalid duration {value!r} (use e.g. 30s, 10m, 14d)")
        total += float(match.group(1)) * _DURATION_SECONDS[match.group(2)]
        pos = match.end()
    return total


def parse_size(value: object) -> int:
    """Bytes for a size string; plain integers are bytes."""
    if isinstance(value, bool):
        raise ValueError("not a size")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ValueError("not a size")
    match = _SIZE.fullmatch(value.strip())
    factor = _SIZE_FACTOR.get(match.group(2).lower()) if match else None
    if match is None or factor is None:
        raise ValueError(f"invalid size {value!r} (use e.g. 64MiB, 1GB)")
    return int(float(match.group(1)) * factor)
