"""
OpenInference instrumentation → Arize Phoenix Cloud.

Captures ALL ADK spans automatically:
  - LLM calls (prompt, completion, token counts, latency)
  - Tool calls (name, input, output, duration)
  - Agent steps (planning, execution, reflection)
  - MCP tool invocations

Every decision MatchMind makes is inspectable in Phoenix.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.semconv.resource import ResourceAttributes

logger = logging.getLogger("matchmind.tracing")

_tracer_provider: TracerProvider | None = None


def setup_tracing(
    phoenix_api_key: str,
    phoenix_base_url: str,
    project_name: str,
) -> TracerProvider:
    """
    One-time setup. Call at application startup before any agent activity.

    Args:
        phoenix_api_key:  Arize Phoenix Cloud API key
        phoenix_base_url: Phoenix endpoint (default: https://app.phoenix.arize.com)
        project_name:     Project name in Phoenix UI (e.g. "matchmind")

    Returns:
        Configured TracerProvider (also set as global OTel provider)
    """
    global _tracer_provider

    resource = Resource(attributes={
        ResourceAttributes.PROJECT_NAME: project_name,
    })

    tracer_provider = TracerProvider(resource=resource)

    # OTLP exporter → Phoenix Cloud
    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{phoenix_base_url}/v1/traces",
        headers={"api_key": phoenix_api_key},
    )

    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
        )
    )

    # Set as global provider
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument Google ADK
    # This patches ADK internals so every agent span is captured
    # without manual instrumentation in tool code.
    GoogleADKInstrumentor().instrument()

    _tracer_provider = tracer_provider
    logger.info(
        "Phoenix tracing active",
        extra={"project": project_name, "endpoint": phoenix_base_url},
    )
    return tracer_provider


def get_tracer(name: str = "matchmind") -> trace.Tracer:
    """Get a named tracer for manual span creation."""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush all pending spans before shutdown."""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("Tracing provider shut down, all spans flushed.")
