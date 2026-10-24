# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac

from hypothesis import given
from hypothesis import strategies as st

from footnote.domain.webhooks import WebhookEvent, backoff_seconds, sign, verify_signature

_SECRET = b"secret"
_BODY = b'{"a":1}'
_TS = 1700000000
_EXPECTED = "sha256=49f24e537407743fa4a0242bb63b94b9a47ee99cbbe071ccd8a22550ae411686"


def test_webhook_event_values_are_dotted_names() -> None:
    assert WebhookEvent.DOCUMENT_READY == "document.ready"
    assert WebhookEvent.DOCUMENT_FAILED == "document.failed"
    assert {e.value for e in WebhookEvent} == {"document.ready", "document.failed"}


def test_sign_known_literal_vector() -> None:
    assert sign(_SECRET, _TS, _BODY) == _EXPECTED


@given(st.binary(), st.integers(min_value=0, max_value=2**40), st.binary())
def test_sign_matches_independent_hmac(secret: bytes, ts: int, body: bytes) -> None:
    digest = hmac.new(secret, str(ts).encode() + b"." + body, hashlib.sha256).hexdigest()
    assert sign(secret, ts, body) == f"sha256={digest}"


def test_verify_signature_accepts_valid_header() -> None:
    assert verify_signature(_SECRET, _TS, _BODY, _EXPECTED) is True


def test_verify_signature_rejects_tampering() -> None:
    assert verify_signature(b"other", _TS, _BODY, _EXPECTED) is False
    assert verify_signature(_SECRET, _TS + 1, _BODY, _EXPECTED) is False
    assert verify_signature(_SECRET, _TS, _BODY + b" ", _EXPECTED) is False
    assert verify_signature(_SECRET, _TS, _BODY, _EXPECTED[len("sha256=") :]) is False
    assert verify_signature(_SECRET, _TS, _BODY, "") is False
    assert verify_signature(_SECRET, _TS, _BODY, "sha256=é") is False


def test_backoff_seconds_grows_exponentially_with_no_jitter_reduction() -> None:
    assert backoff_seconds(1, 2.0, 100.0, 0.5) == 1.0
    assert backoff_seconds(2, 2.0, 100.0, 0.5) == 2.0
    assert backoff_seconds(3, 2.0, 100.0, 0.5) == 4.0
    assert backoff_seconds(3, 2.0, 100.0, 0.0) == 0.0


def test_backoff_seconds_is_capped() -> None:
    assert backoff_seconds(10, 2.0, 30.0, 0.5) == 15.0
    assert backoff_seconds(4, 2.0, 16.0, 0.25) == 4.0


@given(st.integers(min_value=1, max_value=60), st.floats(min_value=0, max_value=0.999))
def test_backoff_seconds_never_exceeds_cap(attempt: int, rand: float) -> None:
    assert 0.0 <= backoff_seconds(attempt, 1.5, 60.0, rand) <= 60.0
