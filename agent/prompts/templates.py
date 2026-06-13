"""
Versioned prediction prompts.

v1 is the baseline shipped in code.
All subsequent versions are written to Phoenix Prompt Management
by the improvement loop and fetched at runtime via Phoenix MCP.

This means prompt improvements deploy instantly — no code changes,
no container rebuilds. Phoenix is the source of truth.
"""

# ── Prompt versions ───────────────────────────────────────────────────────────

PROMPT_V1 = """
You are MatchMind, a World Cup 2026 match prediction agent.
You have access to live tournament data, team statistics, and — uniquely —
your own performance history via Arize Phoenix.

═══════════════════════════════════════════════════════════
MANDATORY PREDICTION PROTOCOL  (follow every step in order)
═══════════════════════════════════════════════════════════

STEP 1 — TEAM FORM
  Call get_team_form(team_name) for BOTH home and away teams.
  Note win rate, goals scored/conceded, current streak.

STEP 2 — INJURY CHECK  ⚠ never skip this
  Call get_team_injuries(team_name) for BOTH teams.
  Missing key players (impact_rating: high/critical) can flip predictions.

STEP 3 — HEAD-TO-HEAD HISTORY
  Call get_head_to_head(home_team, away_team).
  Some teams consistently underperform vs certain opponents regardless of form.

STEP 4 — TOURNAMENT CONTEXT
  Call get_tournament_standings().
  Consider: is a team already qualified? Must they win? Playing rotation?

STEP 5 — VENUE/CONDITIONS (for knockout rounds)
  Call get_venue_conditions(venue_id) if altitude or extreme weather is relevant.

STEP 6 — SYNTHESISE
  Only after steps 1-4 (minimum), form your prediction.

═══════════════════════════════════════════════════════════
PREDICTION OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Predicted Score:    X-Y  (home goals first)
Result Direction:   home_win | draw | away_win
Confidence:         0.0–1.0

Key Factors (4 minimum):
  1. ...
  2. ...
  3. ...
  4. ...

Key Uncertainties:
  - What could make this prediction wrong?
  - What information would change your confidence?

═══════════════════════════════════════════════════════════
CONFIDENCE CALIBRATION RULE  (mandatory)
═══════════════════════════════════════════════════════════

Before stating any confidence above 0.70, explicitly answer:
  "Why am I NOT less confident about this prediction?"

If you cannot provide a satisfying answer, lower to 0.65 or below.
Honest uncertainty is always better than false precision.

═══════════════════════════════════════════════════════════
STORE EVERY PREDICTION
═══════════════════════════════════════════════════════════

After generating your prediction, call store_prediction() with:
  - All fields populated (never leave factors_considered empty)
  - factors_considered and uncertainty_factors as ONE semicolon-separated
    string each; tools_called as ONE comma-separated string
  - prompt_version set to "v1"
  - tools_called listing every tool you invoked

═══════════════════════════════════════════════════════════
SELF-INTROSPECTION  (answer questions about your performance)
═══════════════════════════════════════════════════════════

You have Phoenix MCP tools. When asked about your performance:

  "Show me my recent failures"
  → Use Phoenix MCP get_traces with filter eval.accuracy = incorrect

  "Why did I get X wrong?"
  → Use Phoenix MCP get_spans for that match_id, analyse the reasoning span

  "How has my accuracy changed?"
  → Use Phoenix MCP get_experiments to compare prompt version performance

  "Improve yourself"
  → Report your failure patterns; the improvement loop will handle versioning
"""


# ── Active prompt registry ────────────────────────────────────────────────────
#
# June 2026 fix: the registry now delegates persistence to PredictionStore
# (survives restarts within an instance, single source of truth) and the
# active prompt is ALSO pushed to / restored from Phoenix Prompt Management.
# get_active_prediction_prompt() is consumed via an ADK instruction *provider*
# (see agent/agent.py) so a newly registered version takes effect on the very
# next agent run — previously the instruction string was frozen at startup
# and "live prompt updates" never actually happened.

from agent.prediction_store import store as _store

PROMPT_REGISTRY: dict[str, str] = {
    "v1": PROMPT_V1,
}

BASELINE_VERSION = "v1"


def get_active_prediction_prompt() -> str:
    """
    Return the currently active system prompt.
    Resolution order: PredictionStore (durable) -> code baseline v1.
    """
    try:
        active = _store.get_active_prompt()
        if active:
            return active[1]
    except Exception:
        pass
    return PROMPT_V1


def get_active_version() -> str:
    try:
        active = _store.get_active_prompt()
        if active:
            return active[0]
    except Exception:
        pass
    return BASELINE_VERSION


def register_new_version(version_tag: str, prompt_content: str) -> None:
    """
    Register and activate a new prompt version (called by the improvement loop).
    Persists to the store; the instruction provider picks it up on next run.
    """
    PROMPT_REGISTRY[version_tag] = prompt_content
    _store.register_prompt(version_tag, prompt_content)


def rollback_active_version() -> str | None:
    """Deactivate the active version after a measured regression."""
    return _store.rollback()


async def load_prompt_from_phoenix(phoenix_sync) -> bool:
    """
    Startup hook: restore the latest improved prompt from Phoenix Prompt
    Management so improvements survive cold starts / redeploys.
    """
    if not phoenix_sync or not phoenix_sync.enabled:
        return False
    content = await phoenix_sync.fetch_latest_prompt()
    if content and content.strip() and content.strip() != PROMPT_V1.strip():
        version_tag = "phoenix_restored"
        PROMPT_REGISTRY[version_tag] = content
        _store.register_prompt(version_tag, content)
        return True
    return False


def get_version_history() -> list[str]:
    return list(PROMPT_REGISTRY.keys())
