"""
MatchMind ADK Agent definition.

Wires together:
  - Gemini LLM (required by hackathon rules)
  - Google ADK LlmAgent runtime (required for OpenInference tracing)
  - WC26 MCPToolset  → live World Cup data (18 tools)
  - Phoenix MCPToolset → self-introspection at runtime
  - Custom FunctionTools → prediction storage + match data helpers
"""
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

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


def build_agent() -> LlmAgent:
    """
    Construct and return the fully wired MatchMind agent.
    Call once at application startup after tracing is configured.
    """

    # ── MCP: Phoenix self-introspection ───────────────────────────────────────
    # The agent can query its OWN traces, spans, prompts, and experiments
    # at runtime — this is the self-improvement mechanism.
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
        # Expose only the tools the agent needs for self-introspection
        # (prevents Gemini's context from being flooded with unused tools)
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

    # ── MCP: WC26 live World Cup data ─────────────────────────────────────────
    # Provides 18 live tools: matches, teams, venues, odds, standings,
    # fan zones, city guides, head-to-head records, injuries, news.
    # No API key required.
    wc26_mcp = MCPToolset(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "wc26-mcp@latest"],
        ),
    )

    # ── Custom Python tools ───────────────────────────────────────────────────
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

    # ── Agent assembly ────────────────────────────────────────────────────────
    agent = LlmAgent(
        model=config.GEMINI_MODEL,
        name="matchmind",
        description=(
            "Self-improving World Cup 2026 match prediction agent. "
            "Uses live tournament data and introspects its own prediction history "
            "via Arize Phoenix to continuously improve accuracy."
        ),
        instruction=get_active_prediction_prompt(),
        tools=custom_tools + [phoenix_mcp, wc26_mcp],
    )

    logger.info(
        "MatchMind agent built",
        extra={
            "model": config.GEMINI_MODEL,
            "custom_tools": len(custom_tools),
            "mcp_servers": 2,
        },
    )
    return agent
