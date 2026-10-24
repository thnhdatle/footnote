# SPDX-License-Identifier: Apache-2.0
"""Cursor pagination."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Sequence
from typing import Annotated
from uuid import UUID

from fastapi import Query

from footnote.domain.errors import ValidationError

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
_PREFIX = "c1:"

Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT, description="Page size (1-500).")]
Cursor = Annotated[
    str | None, Query(max_length=128, description="`next_cursor` of the previous page.")
]


def encode_cursor(last_id: UUID | int) -> str:
    """Encodes a cursor that points after an id."""
    raw = (_PREFIX + str(last_id)).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode[T](cursor: str, parse: Callable[[str], T]) -> T:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        if not raw.startswith(_PREFIX):
            raise ValueError(cursor)
        return parse(raw[len(_PREFIX) :])
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValidationError("cursor", "malformed cursor") from exc


def decode_cursor(cursor: str | None) -> UUID | None:
    """Decodes a cursor into the id it points after."""
    return _decode(cursor, UUID) if cursor else None


def decode_int_cursor(cursor: str | None) -> int | None:
    return _decode(cursor, _positive_int) if cursor else None


def _positive_int(text: str) -> int:
    if not text.isdigit():
        raise ValueError(text)
    return int(text)


def next_cursor(ids: Sequence[UUID | int], limit: int) -> str | None:
    if len(ids) <= limit:
        return None
    return encode_cursor(ids[limit - 1])
