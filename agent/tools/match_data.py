"""
Match data tools — WC26 live data via MCP.

Each function is registered as an ADK FunctionTool.
OpenInference auto-instruments all tool calls; we add custom
matchmind.* span attributes for richer Phoenix filtering.
"""
import json
import logging
from datetime import datetime, timedelta
from opentelemetry import trace

tracer = trace.get_tracer("matchmind.tools.match_data")
logger = logging.getLogger("matchmind.tools.match_data")


# ── World Cup schedule data ───────────────────────────────────────────────────
# In production these call the WC26 MCP server tools.
# The MCPToolset in agent.py provides them automatically via ADK.
# These Python wrappers are here so ADK can inspect signatures
# and generate tool descriptions for Gemini.


async def get_upcoming_matches(days_ahead: int = 3) -> dict:
    """
    Get World Cup matches scheduled in the next N days.

    Returns each match with: match_id, home_team, away_team,
    venue, city, date, stage (group/round-of-16/quarter/semi/final).

    Args:
        days_ahead: How many days into the future to look (1-14)
    """
    with tracer.start_as_current_span("tool.get_upcoming_matches") as span:
        span.set_attribute("tool.days_ahead", days_ahead)
        # WC26 MCP provides this at runtime via the MCPToolset
        # Stub return for local dev / unit tests
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)
        result = {
            "matches": [],
            "query_date": datetime.utcnow().isoformat(),
            "cutoff_date": cutoff.isoformat(),
            "source": "wc26_mcp",
        }
        span.set_attribute("tool.match_count", len(result["matches"]))
        return result


async def get_team_form(team_name: str, last_n: int = 5) -> dict:
    """
    Retrieve a team's recent match results and performance stats.

    Returns: results list (W/D/L), goals scored/conceded per game,
    clean sheets, scoring streaks, xG if available.

    Args:
        team_name: Full team name (e.g. "Brazil", "France")
        last_n:    Number of recent matches to analyse (default 5)
    """
    with tracer.start_as_current_span("tool.get_team_form") as span:
        span.set_attribute("tool.team", team_name)
        span.set_attribute("tool.last_n", last_n)
        result = {
            "team": team_name,
            "recent_results": [],          # list of "W"/"D"/"L"
            "goals_scored_avg": 0.0,
            "goals_conceded_avg": 0.0,
            "clean_sheets": 0,
            "win_rate": 0.0,
            "current_streak": None,        # e.g. "3W" or "1L"
            "source": "wc26_mcp",
        }
        span.set_attribute("tool.win_rate", result["win_rate"])
        return result


async def get_head_to_head(
    home_team: str,
    away_team: str,
    limit: int = 10,
) -> dict:
    """
    Historical head-to-head record between two teams.

    Returns: match count, win/draw/loss breakdown, avg goals,
    last 5 meetings with scores, World Cup-specific record.

    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        limit:     Max historical matches to retrieve
    """
    with tracer.start_as_current_span("tool.get_head_to_head") as span:
        span.set_attribute("tool.home_team", home_team)
        span.set_attribute("tool.away_team", away_team)
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "matches_analysed": 0,
            "home_wins": 0,
            "away_wins": 0,
            "draws": 0,
            "home_goals_avg": 0.0,
            "away_goals_avg": 0.0,
            "last_5": [],
            "world_cup_record": {},
            "source": "wc26_mcp",
        }
        return result


async def get_team_injuries(team_name: str) -> dict:
    """
    Current injury, suspension and fitness concern list for a team.

    KEY SIGNAL: missing star players dramatically shift prediction.
    Always call this before predicting any match.

    Args:
        team_name: Full team name
    """
    with tracer.start_as_current_span("tool.get_team_injuries") as span:
        span.set_attribute("tool.team", team_name)
        result = {
            "team": team_name,
            "confirmed_injuries": [],      # {"player": str, "position": str, "return": str}
            "suspensions": [],
            "doubts": [],                  # 50/50 fitness
            "key_players_missing": [],     # subset of above rated "key"
            "impact_rating": "low",        # low / medium / high / critical
            "source": "wc26_mcp",
        }
        span.set_attribute("tool.missing_key_players", len(result["key_players_missing"]))
        span.set_attribute("tool.injury_impact", result["impact_rating"])
        return result


async def get_match_result(match_id: str) -> dict:
    """
    Retrieve the actual final result of a completed match.
    Used by the improvement loop after a match ends.

    Args:
        match_id: Match identifier from WC26 schedule
    """
    with tracer.start_as_current_span("tool.get_match_result") as span:
        span.set_attribute("tool.match_id", match_id)
        result = {
            "match_id": match_id,
            "status": "unknown",           # scheduled / live / completed
            "home_goals": None,
            "away_goals": None,
            "result_string": None,         # e.g. "2-1"
            "scorer_summary": [],
            "source": "wc26_mcp",
        }
        span.set_attribute("tool.match_status", result["status"])
        return result


async def get_tournament_standings() -> dict:
    """
    Current group-stage standings and knockout-stage bracket.

    Context matters: a team that has already qualified may rest players.
    A team that must win will play differently.

    Returns group tables + knockout bracket state.
    """
    with tracer.start_as_current_span("tool.get_tournament_standings"):
        return {
            "groups": {},          # group_id → [{"team", "P", "W", "D", "L", "GD", "Pts"}]
            "knockout_bracket": {},
            "source": "wc26_mcp",
        }


async def get_venue_conditions(venue_id: str) -> dict:
    """
    Stadium + weather conditions for a match venue.

    Altitude, surface, weather forecast — factors that affect
    high-press, physical teams vs technical teams.

    Args:
        venue_id: Venue identifier from WC26 schedule
    """
    with tracer.start_as_current_span("tool.get_venue_conditions") as span:
        span.set_attribute("tool.venue_id", venue_id)
        return {
            "venue_id": venue_id,
            "stadium": "",
            "city": "",
            "altitude_m": 0,
            "capacity": 0,
            "surface": "natural_grass",
            "weather_forecast": {
                "temp_c": None,
                "humidity_pct": None,
                "conditions": "unknown",
            },
            "source": "wc26_mcp",
        }
