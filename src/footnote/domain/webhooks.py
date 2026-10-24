# SPDX-License-Identifier: Apache-2.0
"""Webhook events, payload signing and retry backoff."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import StrEnum

_PREFIX = "sha256="


class WebhookEvent(StrEnum):
    """Events a webhook subscription can receive."""

    DOCUMENT_READY = "document.ready"
    DOCUMENT_FAILED = "document.failed"


ALL_EVENTS: tuple[WebhookEvent, ...] = (WebhookEvent.DOCUMENT_READY, WebhookEvent.DOCUMENT_FAILED)


@dataclass(frozen=True, slots=True)
class WebhookSettings:
    events: tuple[WebhookEvent, ...] = ALL_EVENTS

    def wants(self, event: WebhookEvent) -> bool:
        return event in self.events


def sign(secret: bytes, timestamp: int, body: bytes) -> str:
    """Returns the signature of a timestamp and body."""
    mac = hmac.new(secret, f"{timestamp}.".encode() + body, hashlib.sha256)
    return _PREFIX + mac.hexdigest()


def verify_signature(secret: bytes, timestamp: int, body: bytes, header: str) -> bool:
    return hmac.compare_digest(sign(secret, timestamp, body).encode(), header.encode())


def backoff_seconds(attempt: int, base: float, cap: float, rand: float) -> float:
    ceiling: float = min(cap, base * 2 ** (attempt - 1))
    return rand * ceiling
