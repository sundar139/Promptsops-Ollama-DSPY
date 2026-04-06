from __future__ import annotations

import os

from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")


def start_tracing(endpoint: str | None = None) -> TracerProvider:
    """Configure DSPy OpenInference tracing with OTLP export to Phoenix."""
    otlp_endpoint = endpoint or PHOENIX_ENDPOINT

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    DSPyInstrumentor().instrument()

    return provider


def is_tracing_enabled() -> bool:
    return os.getenv("ENABLE_TRACING", "").lower() in ("1", "true", "yes")
