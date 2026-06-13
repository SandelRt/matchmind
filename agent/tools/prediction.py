"""
Prediction storage tools.

store_prediction() is the most important span in the system — and, since the
June 2026 fix, every prediction is ALSO written to the durable
PredictionStore keyed by match_id (with the span_id captured for later
annotation).

update_prediction_with_result() now actually closes the loop:
  1. records the result against the stored prediction
  2. runs the evaluators immediately (in-process, deterministic)
  3. uploads eval results to Phoenix as SPAN ANNOTATIONS on the original
     prediction span — the correct mechanism, since OTel spans are
     immutable once exported (the old code opened a *new* span and set
     "actual result" attributes there, leaving every prediction span
     permanently 'pending').
"""
import json
import logging
from datetime import datetime, timezone
from opentelemetry import trace

from agent.prediction_store import store

tracer = trace.get_tracer("matchmind.tools.prediction")
logger = logging.getLogger("matchmind.tools.prediction")


def _to_list(value) -> list:
    """
    Normalize a tool argument that may arrive as a JSON-array string, a
    delimited string (";", "|", or ","), or an actual list.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    for delim in (";", "|", ","):
        if delim in s:
            return [p.strip() for p in s.split(delim) if p.strip()]
    return [s]


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
    Persist a match prediction as a richly attributed trace span AND a durable
    store record.

    Args:
        match_id:            WC26 match identifier
        home_team:           Home team name
        away_team:           Away team name
        predicted_score:     e.g. "2-1"
        result_direction:    "home_win" | "away_win" | "draw"
        confidence:          0.0-1.0
        reasoning:           Full reasoning text
        factors_considered:  Semicolon-separated factor strings used in analysis
        uncertainty_factors: Semicolon-separated things that could change outcome
        tools_called:        Comma-separated tool names invoked during analysis
        prompt_version:      Active prompt version tag (e.g. "v1")
    """
    with tracer.start_as_current_span("prediction.store") as span:
        ts = datetime.now(timezone.utc).isoformat()
        span_id = format(span.get_span_context().span_id, "016x")
        trace_id = format(span.get_span_context().trace_id, "032x")

        # Gemini function-calling schemas reject bare `list` params (400
        # INVALID_ARGUMENT — origin fix 7790500), so the tool accepts strings
        # and normalizes here. Internals (store, evaluators, analyzer)
        # continue to work with real lists.
        factors_list = _to_list(factors_considered)
        uncertainty_list = _to_list(uncertainty_factors)
        tools_list = _to_list(tools_called)

        # ── Span attributes (observability) ───────────────────────────────────
        span.set_attribute("matchmind.match_id",            match_id)
        span.set_attribute("matchmind.home_team",           home_team)
        span.set_attribute("matchmind.away_team",           away_team)
        span.set_attribute("matchmind.prediction",          predicted_score)
        span.set_attribute("matchmind.result_direction",    result_direction)
        span.set_attribute("matchmind.confidence",          confidence)
        span.set_attribute("matchmind.timestamp",           ts)
        span.set_attribute("matchmind.prompt_version",      prompt_version)
        span.set_attribute("matchmind.reasoning",           reasoning)
        span.set_attribute("matchmind.factors_considered",  json.dumps(factors_list))
        span.set_attribute("matchmind.uncertainty_factors", json.dumps(uncertainty_list))
        span.set_attribute("matchmind.tools_called",        ",".join(tools_list))
        span.set_attribute("matchmind.tool_count",          len(tools_list))

        # ── Durable record (source of truth for evals + improvement loop) ─────
        store.record_prediction(match_id, {
            "trace_id":         trace_id,
            "span_id":          span_id,
            "match_id":         match_id,
            "home_team":        home_team,
            "away_team":        away_team,
            "prediction":       predicted_score,
            "result_direction": result_direction,
            "confidence":       float(confidence),
            "reasoning":        reasoning,
            "factors":          factors_list,
            "uncertainty":      uncertainty_list,
            "tools_called":     tools_list,
            "tool_count":       len(tools_list),
            "prompt_version":   prompt_version,
            "stored_at":        ts,
        })

        logger.info(
            "Prediction stored",
            extra={"match_id": match_id, "prediction": predicted_score,
                   "confidence": confidence, "span_id": span_id},
        )

        return {
            "status":        "stored",
            "match_id":      match_id,
            "prediction":    predicted_score,
            "confidence":    confidence,
            "trace_span_id": span_id,
        }


async def update_prediction_with_result(
    match_id: str,
    actual_score: str,
    home_goals: int,
    away_goals: int,
) -> dict:
    """
    Attach the actual match result to the stored prediction, run the
    evaluation pipeline, and upload eval annotations to Phoenix.

    Args:
        match_id:     Match identifier (must match stored prediction)
        actual_score: e.g. "3-1"
        home_goals:   Home team final goals
        away_goals:   Away team final goals
    """
    result = await process_match_result(match_id, home_goals, away_goals)
    return result


async def process_match_result(match_id: str, home_goals: int, away_goals: int) -> dict:
    """
    Core result-processing pipeline. Called by the agent tool above AND
    directly by the API as a fallback (so evaluation never depends on the
    LLM remembering to call a tool).
    """
    with tracer.start_as_current_span("prediction.update_result") as span:
        actual_score = f"{home_goals}-{away_goals}"
        direction = (
            "home_win" if home_goals > away_goals
            else "away_win" if away_goals > home_goals
            else "draw"
        )

        span.set_attribute("matchmind.match_id",          match_id)
        span.set_attribute("matchmind.actual_result",     actual_score)
        span.set_attribute("matchmind.actual_direction",  direction)
        span.set_attribute("matchmind.result_updated_at", datetime.now(timezone.utc).isoformat())

        rec = store.record_result(match_id, actual_score, direction)
        if not rec:
            return {"match_id": match_id, "status": "no_stored_prediction",
                    "actual_result": actual_score}

        if rec.get("evaluated"):
            return {"match_id": match_id, "status": "already_evaluated",
                    "actual_result": actual_score, "accuracy": rec.get("accuracy")}

        # ── Evaluate immediately (deterministic, no LLM cost) ─────────────────
        from observability.evaluators import MatchPredictionEvaluator
        evaluator = MatchPredictionEvaluator()
        eval_result = evaluator.evaluate({
            "trace_id": rec.get("trace_id", "unknown"),
            "root_span": {"attributes": {
                "matchmind.prediction":         rec.get("prediction", ""),
                "matchmind.actual_result":      actual_score,
                "matchmind.confidence":         rec.get("confidence", 0.5),
                "matchmind.reasoning":          rec.get("reasoning", ""),
                "matchmind.tools_called":       ",".join(rec.get("tools_called", [])),
                "matchmind.factors_considered": json.dumps(rec.get("factors", [])),
                "matchmind.match_id":           match_id,
                "matchmind.prompt_version":     rec.get("prompt_version", "v1"),
            }},
        })

        eval_dict = {
            "accuracy":           eval_result.accuracy,
            "accuracy_score":     eval_result.accuracy_score,
            "calibration":        eval_result.calibration,
            "calibration_score":  eval_result.calibration_score,
            "reasoning_quality":  eval_result.reasoning_quality,
            "reasoning_score":    eval_result.reasoning_score,
            "composite_score":    eval_result.composite_score,
        }
        store.attach_eval(match_id, eval_dict)

        span.set_attribute("eval.accuracy",  eval_result.accuracy)
        span.set_attribute("eval.composite", eval_result.composite_score)

        # ── Best-effort: annotate the ORIGINAL prediction span in Phoenix ─────
        annotated = False
        try:
            from observability.phoenix_client import get_sync
            sync = get_sync()
            if sync and sync.enabled and rec.get("span_id"):
                annotated = await sync.log_eval_annotations(rec["span_id"], eval_dict)
        except Exception as exc:
            logger.warning("Annotation upload skipped: %s", exc)

        logger.info(
            "Result evaluated",
            extra={"match_id": match_id, "actual": actual_score,
                   "accuracy": eval_result.accuracy, "annotated": annotated},
        )

        return {
            "match_id":            match_id,
            "actual_result":       actual_score,
            "actual_direction":    direction,
            "accuracy":            eval_result.accuracy,
            "calibration":         eval_result.calibration,
            "reasoning_quality":   eval_result.reasoning_quality,
            "composite_score":     eval_result.composite_score,
            "phoenix_annotated":   annotated,
            "eval_triggered":      True,
        }
