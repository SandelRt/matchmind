"""
OpenInference instrumentation → Arize Phoenix Cloud.

Captures ALL spans automatically across two instrumentors:

  GoogleADKInstrumentor (agent runtime):
  - LLM calls (prompt, completion, token counts, latency)
  - Tool calls (name, input, output, duration)
  - Agent steps (planning, execution, reflection)
  - MCP tool invocations (Phoenix MCP + WC26 MCP)

  GoogleGenAIInstrumentor (direct Gemini calls):
  - Self-improvement meta-prompting (SelfImprovementLoop)
  - Gemini prompt rewrite calls (before/after prompt versions)
  - Any google-genai SDK calls outside ADK context

Together these give complete end-to-end tracing: every prediction
decision AND every self-improvement cycle is visible in Phoenix.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
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
    # Patches ADK internals — every agent span captured without
    # manual instrumentation in tool code.
    GoogleADKInstrumentor().instrument()

    # Auto-instrument google-genai SDK
    # Captures direct Gemini calls made by the self-improvement loop
    # (SelfImprovementL


def shutdown_tracing() -> None:
    """Flush and shut down the global TracerProvider."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
