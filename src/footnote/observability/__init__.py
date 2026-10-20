# SPDX-License-Identifier: Apache-2.0
"""Observability: structlog logging, Prometheus metrics, OpenTelemetry tracing.

Main types: `make_logger` (fixed key vocabulary, JSON or text), `Metrics` (one registry per
process, passed as a dependency), `setup_tracing` / `tracer` (off unless OTLP env is set).

    log = make_logger(settings.log, component="worker")
    log.info("job done", job_id=str(job.id), duration_ms=12)
"""

from footnote.observability.logging import (
    ALLOWED_KEYS,
    CONTENT_LIMIT,
    capture_content,
    make_logger,
    unknown_keys,
)
from footnote.observability.metrics import Metrics
from footnote.observability.tracing import (
    Tracing,
    inject_traceparent,
    set_chat_attributes,
    setup_tracing,
    tracer,
    tracing_enabled,
)

__all__ = [
    "ALLOWED_KEYS",
    "CONTENT_LIMIT",
    "Metrics",
    "Tracing",
    "capture_content",
    "inject_traceparent",
    "make_logger",
    "set_chat_attributes",
    "setup_tracing",
    "tracer",
    "tracing_enabled",
    "unknown_keys",
]
