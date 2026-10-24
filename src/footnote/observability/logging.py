# SPDX-License-Identifier: Apache-2.0
"""Structured logging with JSON or text output."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any, TextIO

import structlog
from opentelemetry import trace
from structlog.typing import EventDict, Processor, WrappedLogger

from footnote.config import LogConfig

CONTENT_LIMIT = 256

BASE_KEYS = frozenset({"ts", "level", "msg"})
ALLOWED_KEYS = BASE_KEYS | {
    "request_id",
    "trace_id",
    "span_id",
    "component",
    "operation",
    "duration_ms",
    "status",
    "error",
    "provider",
    "model",
    "tenant",
    "job_id",
    "queue",
    "attempt",
    "table",
    "lsn",
    "key",
    "count",
    "bytes",
    "collection_id",
    "document_id",
}

_SECRET_MARKERS = ("password", "secret", "api_key", "apikey", "authorization", "token", "cookie")
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def unknown_keys(event: Mapping[str, Any]) -> set[str]:
    return set(event) - ALLOWED_KEYS


def capture_content(text: str, *, enabled: bool) -> str | None:
    if not enabled:
        return None
    return text if len(text) <= CONTENT_LIMIT else text[:CONTENT_LIMIT]


def _redact(_: WrappedLogger, __: str, event: MutableMapping[str, Any]) -> EventDict:
    for key in event:
        lowered = key.lower()
        if key not in ALLOWED_KEYS and any(m in lowered for m in _SECRET_MARKERS):
            event[key] = "[redacted]"
    return event


def _trace_ids(_: WrappedLogger, __: str, event: MutableMapping[str, Any]) -> EventDict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event.setdefault("trace_id", f"{ctx.trace_id:032x}")
        event.setdefault("span_id", f"{ctx.span_id:016x}")
    return event


def _strict(_: WrappedLogger, __: str, event: MutableMapping[str, Any]) -> EventDict:
    extra = unknown_keys(event)
    if extra:
        raise ValueError(f"log keys outside the vocabulary: {sorted(extra)}")
    return event


def make_logger(
    config: LogConfig,
    *,
    component: str,
    stream: TextIO | None = None,
    strict: bool = False,
) -> structlog.typing.FilteringBoundLogger:
    """Builds a logger for a component."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _trace_ids,
        _redact,
        structlog.processors.format_exc_info,
        structlog.processors.EventRenamer("msg"),
    ]
    if strict:
        processors.append(_strict)
    if config.format == "text":
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    else:
        processors.append(structlog.processors.JSONRenderer())
    return structlog.wrap_logger(  # type: ignore[no-any-return]
        structlog.PrintLogger(file=stream or sys.stderr),
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[config.level]),
        cache_logger_on_first_use=False,
    ).bind(component=component)
