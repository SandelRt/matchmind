"""
Prediction storage tools.

store_prediction() is the most important span in the system.
Everything the evaluators and improvement loop need lives here
as structured span attributes — queryable in Phoenix.
"""
import json
import logging
from datetime import datetime, timezone
from opentelemetry import trace

tracer = trace.get_tracer("matchmind.tools.prediction")
logger = logging.getLogger("matchmind.tools.prediction")


async def store_prediction(
    match_id: str,
    home_team: str,
    away_team: str,
    predicted_score: str,
    result_direction: str,
    confidence: float,
    reasoning: str,
    factors_considered: str,
    uncertainty_factors: str,
    tools_called: str,
    prompt_version: str,
) -> dict:
    """
    Persist a match prediction as a richly attributed trace span.

    CRITICAL: this span is the primary artifact for the eval pipeline.
    All matchmind.* attributes are indexed in Phoenix and filterable
    by the improvement loop and the Phoenix MCP queries.

    Args:
        match_id:            WC26 match identifier
        home_team:           Home team name
        away_team:           Away team name
        predicted_score:     e.g. "2-1"
        result_direction:    "home_win" | "away_win" | "draw"
        confidence:          0.0–1.0
        reasoning:           Full reasoning text
        factors_considered:  List of factor strings used in analysis
        uncertainty_factors: List of things that could change outcome
        tools_called:        List of tool names invoked during analysis
        prompt_version:      Active prompt version tag (e.g. "v1", "v20260611_0930")
    """
    with tracer.start_as_current_span("prediction.store") as span:
        ts = datetime.now(timezone.utc).isoformat()

        # ── Primary prediction attributes ─────────────────────────────────────
        span.set_attribute("matchmind.match_id",          match_id)
        span.set_attribute("matchmind.home_team",         home_team)
        span.set_attribute("matchmind.away_team",         away_team)
        span.set_attribute("matchmind.prediction",        predicted_score)
        span.set_attribute("matchmind.result_direction",  result_direction)
        span.set_attribute("matchmind.confidence",        confidence)
        span.set_attribute("matchmind.timestamp",         ts)
        span.set_attribute("matchmind.prompt_version",    prompt_version)

        # ── Rich reasoning context ────────────────────────────────────────────
        span.set_attribute("matchmind.reasoning",              reasoning)
        span.set_attribute("matchmind.factors_considered",     factors_considered if isinstance(factors_considered, str) else json.dumps(factors_considered))
        span.set_attribute("matchmind.uncertainty_factors",    uncertainty_factors if isinstance(uncertainty_factors, str) else json.dumps(uncertainty_factors))
        span.set_attribute("matchmind.tools_called",           tools_called if isinstance(tools_called, str) else ",".join(tools_called))
        span.set_attribute("matchmind.tool_count",             len(tools_called.split(",")) if isinstance(tools_called, str) else len(tools_called))

        # ── Placeholder for actual result (filled by update_prediction_with_result) ─
        span.set_attribute("matchmind.actual_result",   "pending")
        span.set_attribute("matchmind.evaluated",       False)

        prediction_record = {
            "match_id":           match_id,
            "home_team":          home_team,
            "away_team":          away_team,
            "prediction":         predicted_score,
            "result_direction":   result_direction,
            "confidence":         confidence,
            "reasoning":          reasoning,
            "factors_considered": factors_considered,
            "uncertainty_factors":uncertainty_factors,
            "tools_called":       tools_called,
            "prompt_version":     prompt_version,
            "stored_at":          ts,
        }

        logger.info(
            "Prediction stored",
            extra={"match_id": match_id, "prediction": predicted_score, "confidence": confidence},
        )

        return {
            "status":        "stored",
            "prediction":    prediction_record,
            "trace_span_id": format(span.get_span_context().span_id, "016x"),
        }


async def update_prediction_with_result(
    match_id: str,
    actual_score: str,
    home_goals: int,
    away_goals: int,
) -> dict:
    """
    Attach the actual match result to the stored prediction trace.
    Calling this triggers the downstream evaluation pipeline.

    Args:
        match_id:    Match identifier (must match stored prediction)
        actual_score: e.g. "3-1"
        home_goals:  Home team final goals
        away_goals:  Away team final goals
    """
    with tracer.start_as_current_span("prediction.update_result") as span:
        direction = (
            "home_win" if home_goals > away_goals
            else "away_win" if away_goals > home_goals
            else "draw"
        )

        span.set_attribute("matchmind.match_id",       match_id)
        span.set_attribute("matchmind.actual_result",  actual_score)
        span.set_attribute("matchmind.actual_home_goals", home_goals)
        span.set_attribute("matchmind.actual_away_goals", away_goals)
        span.set_attribute("matchmind.actual_direction",  direction)
        span.set_attribute("matchmind.result_updated_at", datetime.now(timezone.utc).isoformat())

        logger.info(
            "Result updated",
            extra={"match_id": match_id, "actual": actual_score},
        )

        return {
            "match_id":        match_id,
            "actual_result":   actual_score,
            "actual_direction": direction,
            "eval_triggered":  True,
        }
