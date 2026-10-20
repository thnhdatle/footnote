# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry tracing: off unless an OTLP endpoint is configured.

Configuration is the standard `OTEL_*` environment: tracing turns on when
`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set and
`OTEL_SDK_DISABLED` is not true. Exporter options (headers, protocol endpoint, timeout)
are read from `OTEL_EXPORTER_OTLP_*` by the exporter itself.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

TRACER_NAME = "footnote"
_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
_propagator = TraceContextTextMapPropagator()


def tracing_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the environment asks for trace export."""
    source = os.environ if env is None else env
    if source.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    return any(source.get(name, "").strip() for name in _ENDPOINT_VARS)


@dataclass(frozen=True, slots=True)
class Tracing:
    """Handle returned by `setup_tracing`; `shutdown` flushes pending spans."""

    provider: TracerProvider | None

    @property
    def enabled(self) -> bool:
        """True when spans are exported."""
        return self.provider is not None

    def tracer(self) -> trace.Tracer:
        """Tracer of this provider (no-op tracer when disabled)."""
        if self.provider is None:
            return trace.NoOpTracer()
        return self.provider.get_tracer(TRACER_NAME)

    def shutdown(self) -> None:
        """Flush and stop the exporter."""
        if self.provider is not None:
            self.provider.shutdown()


def setup_tracing(
    *,
    version: str,
    env: Mapping[str, str] | None = None,
    exporter: SpanExporter | None = None,
    install: bool = True,
) -> Tracing:
    """Build the provider when enabled (or when `exporter` is given, for tests).

    `install` registers it as the process-wide provider (OpenTelemetry allows this once).
    """
    if exporter is None and not tracing_enabled(env):
        return Tracing(None)
    provider = TracerProvider(
        resource=Resource.create({"service.name": "footnote", "service.version": version})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter or OTLPSpanExporter()))
    if install:
        trace.set_tracer_provider(provider)
    return Tracing(provider)


def tracer() -> trace.Tracer:
    """The process-wide tracer (no-op until `setup_tracing` installs a provider)."""
    return trace.get_tracer(TRACER_NAME)


def inject_traceparent(headers: MutableMapping[str, str]) -> None:
    """Add W3C `traceparent` (and `tracestate`) of the current span to outgoing headers."""
    _propagator.inject(headers)


def set_chat_attributes(  # noqa: PLR0913 - one keyword per GenAI attribute
    span: trace.Span,
    *,
    provider: str,
    request_model: str,
    temperature: float,
    max_tokens: int,
    response_model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """GenAI semantic-convention attributes of a `chat <model>` span."""
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", request_model)
    span.set_attribute("gen_ai.request.temperature", temperature)
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
