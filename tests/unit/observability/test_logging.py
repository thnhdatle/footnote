# SPDX-License-Identifier: Apache-2.0
"""structlog setup: keys, redaction, levels, trace ids."""

from __future__ import annotations

import io
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider

from footnote.config import LogConfig
from footnote.observability import (
    ALLOWED_KEYS,
    CONTENT_LIMIT,
    capture_content,
    make_logger,
    unknown_keys,
)


def lines(buf: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def test_json_line_has_standard_keys() -> None:
    buf = io.StringIO()
    log = make_logger(LogConfig(), component="worker", stream=buf)
    log.info("job done", job_id="j1", duration_ms=12, count=3)
    [event] = lines(buf)
    assert event["msg"] == "job done"
    assert event["level"] == "info"
    assert event["component"] == "worker"
    assert str(event["ts"]).endswith("Z")
    assert unknown_keys(event) == set()


def test_level_filtering() -> None:
    buf = io.StringIO()
    log = make_logger(LogConfig(level="warn"), component="c", stream=buf)
    log.info("hidden")
    log.warning("shown")
    assert [e["msg"] for e in lines(buf)] == ["shown"]


def test_secret_like_keys_are_redacted() -> None:
    buf = io.StringIO()
    log = make_logger(LogConfig(), component="c", stream=buf)
    log.info("x", api_key="sk-1", authorization="Bearer z", key="k1")
    [event] = lines(buf)
    assert event["api_key"] == "[redacted]"
    assert event["authorization"] == "[redacted]"
    assert event["key"] == "k1"


def test_strict_rejects_unknown_keys() -> None:
    log = make_logger(LogConfig(), component="c", stream=io.StringIO(), strict=True)
    log.info("fine", tenant="t1", collection_id="c1", document_id="d1")
    with pytest.raises(ValueError, match="surprise"):
        log.info("bad", surprise=1)


def test_trace_ids_added_inside_a_span() -> None:
    buf = io.StringIO()
    log = make_logger(LogConfig(), component="c", stream=buf)
    log.info("outside")
    with TracerProvider().get_tracer("t").start_as_current_span("s"):
        log.info("inside")
    outside, inside = lines(buf)
    assert "trace_id" not in outside
    assert len(str(inside["trace_id"])) == 32
    assert len(str(inside["span_id"])) == 16


def test_text_format_and_exception() -> None:
    buf = io.StringIO()
    log = make_logger(LogConfig(format="text"), component="c", stream=buf)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    out = buf.getvalue()
    assert "failed" in out
    assert "RuntimeError" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.splitlines()[0])


def test_capture_content_truncates_and_gates() -> None:
    assert capture_content("secret prompt", enabled=False) is None
    assert capture_content("short", enabled=True) == "short"
    assert capture_content("x" * 1000, enabled=True) == "x" * CONTENT_LIMIT


def test_vocabulary_has_standard_and_project_keys() -> None:
    for key in ("request_id", "trace_id", "tenant", "lsn", "bytes", "collection_id", "document_id"):
        assert key in ALLOWED_KEYS
