# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from footnote.observability import (
    inject_traceparent,
    set_chat_attributes,
    setup_tracing,
    tracer,
    tracing_enabled,
)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://loupe:4318"}, True),
        ({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://loupe:4318/v1/traces"}, True),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "  "}, False),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x", "OTEL_SDK_DISABLED": "true"}, False),
    ],
)
def test_tracing_enabled_from_env(env: dict[str, str], expected: bool) -> None:
    assert tracing_enabled(env) is expected


def test_disabled_by_default_is_noop() -> None:
    handle = setup_tracing(version="1", env={}, install=False)
    assert not handle.enabled
    with handle.tracer().start_as_current_span("x") as span:
        assert not span.get_span_context().is_valid
    handle.shutdown()
    assert tracer() is not None


def test_enabled_exports_spans_with_genai_attributes_and_traceparent() -> None:
    exporter = InMemorySpanExporter()
    handle = setup_tracing(version="9.9", env={}, exporter=exporter, install=False)
    assert handle.provider is not None
    headers: dict[str, str] = {}
    with handle.tracer().start_as_current_span("chat m") as span:
        set_chat_attributes(
            span,
            provider="openai_compat",
            request_model="m",
            temperature=0.0,
            max_tokens=512,
            response_model="m-2",
            input_tokens=10,
            output_tokens=3,
        )
        inject_traceparent(headers)
        trace_id = f"{span.get_span_context().trace_id:032x}"
    handle.provider.force_flush()
    handle.shutdown()
    assert headers["traceparent"].split("-")[1] == trace_id
    [finished] = exporter.get_finished_spans()
    attrs = finished.attributes or {}
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai_compat"
    assert attrs["gen_ai.request.model"] == "m"
    assert attrs["gen_ai.response.model"] == "m-2"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert finished.resource.attributes["service.version"] == "9.9"


def test_attributes_skip_unknown_response_fields() -> None:
    exporter = InMemorySpanExporter()
    handle = setup_tracing(version="1", env={}, exporter=exporter, install=False)
    assert handle.provider is not None
    with handle.tracer().start_as_current_span("chat m") as span:
        set_chat_attributes(span, provider="p", request_model="m", temperature=0, max_tokens=1)
    handle.provider.force_flush()
    [finished] = exporter.get_finished_spans()
    assert "gen_ai.response.model" not in (finished.attributes or {})
    assert "gen_ai.usage.input_tokens" not in (finished.attributes or {})
