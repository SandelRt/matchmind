"""
Match data tools — backed by agent/data/wc2026.py static dataset.

Each function is a real implementation (not a stub) that the ADK agent
calls as a FunctionTool. OpenInference auto-instruments all tool calls;
we add matchmind.* span attributes for richer Phoenix filtering.

In a production system these would hit live APIs. For this demo they
read from the embedded WC2026 dataset — which is enough for Gemini to
reason meaningfully about every Group Stage match.
"""
import json
import logging
from datetime import datetime, timedelta
from opentelemetry import trace

from agent.data.wc2026 import (
    get_team_data,
    get_h2h_data,
    get_upcoming,
    get_venue,
    GROUPS,
    get_initial_standings,
    UPCOMING_MATCHES,
    VENUES,
)

tracer = trace.get_tracer("matchmind.tools.match_data")
logger = logging.getLogger("matchmind.tools.match_data")

# In-memory result store: match_id → {"home_goals": int, "away_goals": int}
_results_store: dict[str, dict] = {}


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

        today = datetime.utcnow().date()
        cutoff = today + timedelta(days=days_ahead)

        matches = []
        for m in UPCOMING_MATCHES:
            if m["stage"] not in ("group",):
                continue  # only surface concrete matches
            try:
                match_date = datetime.strptime(m["date"], "%Y-%m-%d").date()
            except ValueError:
                continue
            if today <= match_date <= cutoff:
                venue = VENUES.get(m["venue_id"], {})
                matches.append({
                    "match_id":   m["match_id"],
                    "home_team":  m["home_team"],
                    "away_team":  m["away_team"],
                    "date":       m["date"],
                    "stage":      m["stage"],
                    "group":      m.get("group"),
                    "venue":      venue.get("stadium", ""),
                    "city":       venue.get("city", ""),
                    "venue_id":   m["venue_id"],
                })

        result = {
            "matches": matches,
            "query_date": datetime.utcnow().isoformat(),
            "cutoff_date": cutoff.isoformat(),
            "source": "wc2026_static",
        }
        span.set_attribute("tool.match_count", len(matches))
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

        data = get_team_data(team_name)
        if not data:
            result = {
                "team": team_name,
                "recent_results": [],
                "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0,
                "clean_sheets": 0,
                "win_rate": 0.0,
                "current_streak": None,
                "key_players": [],
                "playing_style": "unknown",
                "notes": f"No form data available for {team_name}",
                "source": "wc2026_static",
            }
        else:
            result = {
                "team": team_name,
                "recent_results": data["recent_results"][:last_n],
                "goals_scored_avg": data["goals_scored_avg"],
                "goals_conceded_avg": data["goals_conceded_avg"],
                "clean_sheets": data["clean_sheets"],
                "win_rate": data["win_rate"],
                "current_streak": data["current_streak"],
                "key_players": data.get("key_players", []),
                "playing_style": data.get("playing_style", ""),
                "notes": data.get("notes", ""),
                "source": "wc2026_static",
            }

        span.set_attribute("tool.win_rate", result["win_rate"])
        span.set_attribute("tool.current_streak", str(result.get("current_streak", "")))
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

        data = get_h2h_data(home_team, away_team)
        if not data:
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
                "notes": f"No H2H data for {home_team} vs {away_team}",
                "source": "wc2026_static",
            }
        else:
            result = {
                "home_team": home_team,
                "away_team": away_team,
                "matches_analysed": data.get("matches_analysed", 0),
                "home_wins": data.get("home_wins", 0),
                "away_wins": data.get("away_wins", 0),
                "draws": data.get("draws", 0),
                "home_goals_avg": data.get("home_goals_avg", 0.0),
                "away_goals_avg": data.get("away_goals_avg", 0.0),
                "last_5": data.get("last_5", []),
                "world_cup_record": data.get("world_cup_record", {}),
                "notes": data.get("notes", ""),
                "source": "wc2026_static",
            }

        span.set_attribute("tool.h2h_matches", result["matches_analysed"])
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

        # Static dataset: assume clean bills of health at tournament start.
        # In production, this would call a live injury API.
        # We surface the team's key players so the agent knows who to watch.
        data = get_team_data(team_name)
        key_players = data.get("key_players", []) if data else []

        result = {
            "team": team_name,
            "confirmed_injuries": [],
            "suspensions": [],
            "doubts": [],
            "key_players_missing": [],
            "impact_rating": "low",
            "key_players_available": key_players,
            "notes": (
                "Static dataset: no injury reports at tournament start. "
                "All key players assumed available. "
                "Update with live data as tournament progresses."
            ),
            "source": "wc2026_static",
        }

        span.set_attribute("tool.missing_key_players", 0)
        span.set_attribute("tool.injury_impact", "low")
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

        stored = _results_store.get(match_id)
        if stored:
            hg = stored["home_goals"]
            ag = stored["away_goals"]
            result = {
                "match_id":     match_id,
                "status":       "completed",
                "home_goals":   hg,
                "away_goals":   ag,
                "result_string": f"{hg}-{ag}",
                "scorer_summary": [],
                "source":       "submitted_result",
            }
        else:
            result = {
                "match_id":     match_id,
                "status":       "scheduled",
                "home_goals":   None,
                "away_goals":   None,
                "result_string": None,
                "scorer_summary": [],
                "source":       "wc2026_static",
            }

        span.set_attribute("tool.match_status", result["status"])
        return result


def record_result(match_id: str, home_goals: int, away_goals: int) -> None:
    """Store a submitted result so get_match_result can return it."""
    _results_store[match_id] = {"home_goals": home_goals, "away_goals": away_goals}


async def get_tournament_standings() -> dict:
    """
    Current group-stage standings and knockout-stage bracket.

    Context matters: a team that has already qualified may rest players.
    A team that must win will play differently.

    Returns group tables + knockout bracket state.
    """
    with tracer.start_as_current_span("tool.get_tournament_standings"):
        # Build standings from any stored results
        standings = get_initial_standings()

        for match_id, result in _results_store.items():
            # Look up the match in schedule
            match = next((m for m in UPCOMING_MATCHES if m["match_id"] == match_id), None)
            if not match or not match.get("group"):
                continue
            group = match["group"]
            home = match["home_team"]
            away = match["away_team"]
            hg = result["home_goals"]
            ag = result["away_goals"]

            group_table = standings.get(group, [])
            home_row = next((r for r in group_table if r["team"] == home), None)
            away_row = next((r for r in group_table if r["team"] == away), None)

            if home_row and away_row:
                home_row["P"] += 1; away_row["P"] += 1
                home_row["GF"] += hg; home_row["GA"] += ag
                away_row["GF"] += ag; away_row["GA"] += hg
                home_row["GD"] = home_row["GF"] - home_row["GA"]
                away_row["GD"] = away_row["GF"] - away_row["GA"]
                if hg > ag:
                    home_row["W"] += 1; home_row["Pts"] += 3
                    away_row["L"] += 1
                elif hg < ag:
                    away_row["W"] += 1; away_row["Pts"] += 3
                    home_row["L"] += 1
                else:
                    home_row["D"] += 1; home_row["Pts"] += 1
                    away_row["D"] += 1; away_row["Pts"] += 1

        # Sort each group by Pts desc, then GD desc
        for group in standings:
            standings[group].sort(key=lambda r: (r["Pts"], r["GD"], r["GF"]), reverse=True)

        return {
            "groups": standings,
            "knockout_bracket": {
                "round_of_32": "TBD",
                "quarter_finals": "TBD",
                "semi_finals": "TBD",
                "final": "TBD",
            },
            "total_results_recorded": len(_results_store),
            "source": "wc2026_static",
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

        venue = get_venue(venue_id)
        if not venue:
            return {
                "venue_id": venue_id,
                "stadium": "Unknown",
                "city": "Unknown",
                "altitude_m": 0,
                "capacity": 0,
                "surface": "natural_grass",
                "altitude_impact": "none",
                "weather_forecast": {
                    "temp_c": 22,
                    "humidity_pct": 50,
                    "conditions": "clear",
                },
                "source": "wc2026_static",
            }

        altitude = venue.get("altitude_m", 0)
        if altitude > 2000:
            altitude_impact = "high — expect reduced stamina for sea-level teams"
        elif altitude > 800:
            altitude_impact = "moderate — slight advantage to acclimatised teams"
        else:
            altitude_impact = "none"

        return {
            "venue_id": venue_id,
            "stadium": venue["stadium"],
            "city": venue["city"],
            "country": venue.get("country", ""),
            "altitude_m": altitude,
            "capacity": venue.get("capacity", 0),
            "surface": venue.get("surface", "natural_grass"),
            "altitude_impact": altitude_impact,
            "weather_forecast": {
                "temp_c": 24,          # static placeholder — live weather in prod
                "humidity_pct": 55,
                "conditions": "clear",
            },
            "source": "wc2026_static",
        }
