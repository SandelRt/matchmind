"""
MatchMind ADK Agent definition.

Wires together:
  - Gemini LLM (Google AI Studio)
  - Google ADK LlmAgent runtime
  - Phoenix MCPToolset → self-introspection at runtime (optional)
  - Custom FunctionTools → all 9 match-data + prediction tools
"""
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from agent.config import config
from agent.prompts.templates import get_active_prediction_prompt
from agent.tools.match_data import (
    get_upcoming_matches,
    get_team_form,
    get_head_to_head,
    get_team_injuries,
    get_match_result,
    get_tournament_standings,
    get_venue_conditions,
)
from agent.tools.prediction import (
    store_prediction,
    update_prediction_with_result,
)

logger = logging.getLogger("matchmind.agent")


def _build_phoenix_mcp():
    """
    Try to build the Phoenix MCP toolset for runtime self-introspection.
    Returns None (and logs a warning) if npx / the package is unavailable —
    the agent degrades gracefully without it.
    """
    if not config.PHOENIX_API_KEY:
        logger.info("PHOENIX_API_KEY not set — Phoenix MCP disabled")
        return None

    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
        phoenix_mcp = MCPToolset(
            connection_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@arizeai/phoenix-mcp@latest",
                    "--baseUrl", config.PHOENIX_BASE_URL,
                    "--apiKey",  config.PHOENIX_API_KEY,
                ],
            ),
            # Expose only the introspection subset — keeps Gemini's tool list lean
            tool_filter=[
                "list_projects",
                "get_traces",
                "get_spans",
                "get_prompts",
                "create_prompt",
                "get_experiments",
                "get_datasets",
                "get_annotation_configs",
                "get_sessions",
            ],
        )
        logger.info("Phoenix MCP toolset configured")
        return phoenix_mcp
    except Exception as exc:
        logger.warning("Phoenix MCP unavailable (npx not found?): %s — running without it", exc)
        return None


def _instruction_provider(context=None) -> str:
    """
    ADK InstructionProvider — called by the runtime before each agent run.
    Always returns the CURRENT active prompt, so versions registered by the
    self-improvement loop go live on the next prediction without redeploy.
    """
    return get_active_prediction_prompt()


def build_agent() -> LlmAgent:
    """
    Construct and return the fully wired MatchMind agent.
    Call once at application startup after tracing is configured.
    """

    # ── Custom Python tools (always available) ────────────────────────────────
    custom_tools = [
        FunctionTool(get_upcoming_matches),
        FunctionTool(get_team_form),
        FunctionTool(get_head_to_head),
        FunctionTool(get_team_injuries),
        FunctionTool(get_match_result),
        FunctionTool(get_tournament_standings),
        FunctionTool(get_venue_conditions),
        FunctionTool(store_prediction),
        FunctionTool(update_prediction_with_result),
    ]

    # ── Optional: Phoenix MCP for runtime self-introspection ──────────────────
    tools = list(custom_tools)
    phoenix_mcp = _build_phoenix_mcp()
    if phoenix_mcp:
        tools.append(phoenix_mcp)

    # ── Agent assembly ────────────────────────────────────────────────────────
    # NOTE (June 2026 fix): `instruction` is an ADK InstructionProvider
    # (callable, re-evaluated on every run) — NOT a frozen string. This is
    # what makes improvement-loop prompt updates take effect immediately
    # without a redeploy. Passing get_active_prediction_prompt() by value
    # here would silently freeze the v1 prompt forever.
    agent = LlmAgent(
        model=config.GEMINI_MODEL,
        name="matchmind",
        description=(
            "Self-improving World Cup 2026 match prediction agent. "
            "Uses live tournament data and introspects its own prediction history "
            "via Arize Phoenix to continuously improve accuracy."
        ),
        instruction=_instruction_provider,
        tools=tools,
    )

    logger.info(
        "MatchMind agent built — model=%s  custom_tools=%d  phoenix_mcp=%s",
        config.GEMINI_MODEL,
        len(custom_tools),
        "enabled" if phoenix_mcp else "disabled",
    )
    return agent
