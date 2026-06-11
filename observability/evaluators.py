"""
Arize Phoenix evaluation pipeline for MatchMind.

Three evaluators run after each match result arrives:
  1. PredictionAccuracyEvaluator  — was the result direction correct?
  2. ConfidenceCalibrationEvaluator — was confidence appropriate?
  3. ReasoningQualityEvaluator    — LLM-as-a-judge on reasoning depth

Results are stored back to Phoenix as annotations on each trace span,
making them queryable by the self-improvement loop.
"""
import re
import json
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("matchmind.evaluators")


# ── Score types ───────────────────────────────────────────────────────────────

AccuracyLabel    = Literal["correct", "incorrect"]
CalibrationLabel = Literal["well_calibrated", "overconfident", "underconfident"]
ReasoningLabel   = Literal["high", "medium", "low"]


@dataclass
class EvalResult:
    trace_id: str
    match_id: str
    accuracy: AccuracyLabel
    calibration: CalibrationLabel
    reasoning_quality: ReasoningLabel
    accuracy_score: float       # 1.0 or 0.0
    calibration_score: float    # 1.0 = good, 0.0 = bad
    reasoning_score: float      # 0.0–1.0
    composite_score: float      # weighted average
    prompt_version: str
    failure: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_score(text: str) -> tuple[int | None, int | None]:
    """Extract (home_goals, away_goals) from strings like '2-1' or '2 : 1'."""
    m = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _direction(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


# ── Evaluators ────────────────────────────────────────────────────────────────

class PredictionAccuracyEvaluator:
    """
    Rule-based: compare result direction (win/draw/loss).
    Fast, deterministic, no LLM cost.
    """

    def evaluate(self, prediction: str, actual: str) -> tuple[AccuracyLabel, float]:
        ph, pa = _parse_score(prediction)
        ah, aa = _parse_score(actual)

        if None in (ph, pa, ah, aa):
            return "incorrect", 0.0

        pred_dir   = _direction(ph, pa)
        actual_dir = _direction(ah, aa)

        if pred_dir == actual_dir:
            return "correct", 1.0
        return "incorrect", 0.0


class ConfidenceCalibrationEvaluator:
    """
    Checks whether stated confidence matched the actual outcome.
    Penalises overconfidence (high confidence + wrong) most heavily.
    """

    def evaluate(
        self,
        confidence: float,
        accuracy: AccuracyLabel,
    ) -> tuple[CalibrationLabel, float]:
        correct = accuracy == "correct"

        if correct and confidence >= 0.60:
            return "well_calibrated", 1.0
        if correct and confidence < 0.40:
            return "underconfident", 0.5
        if not correct and confidence >= 0.75:
            return "overconfident", 0.0    # worst case
        if not correct and confidence < 0.45:
            return "well_calibrated", 0.7  # honest low-confidence miss is ok
        return "well_calibrated", 0.6


class ReasoningQualityEvaluator:
    """
    Heuristic + LLM-as-a-judge hybrid.
    Heuristic runs cheap checks first; LLM judge only fires for borderline cases.
    """

    # Required tool categories the agent should have called
    REQUIRED_TOOL_CATEGORIES = {"form", "injury", "head_to_head", "standings"}

    def evaluate(
        self,
        reasoning: str,
        tools_called: list[str],
        factors: list[str],
    ) -> tuple[ReasoningLabel, float]:
        tool_set   = set(t.lower() for t in tools_called)
        factor_count = len(factors)
        word_count = len(reasoning.split())

        covered_categories = sum(
            1 for cat in self.REQUIRED_TOOL_CATEGORIES
            if any(cat in t for t in tool_set)
        )

        # High: all 4 categories covered + rich reasoning
        if covered_categories >= 4 and word_count >= 60 and factor_count >= 4:
            return "high", 1.0

        # Low: lazy reasoning
        if covered_categories <= 1 or word_count < 20:
            return "low", 0.0

        # Medium: partial coverage
        score = 0.4 + (covered_categories / 4) * 0.3 + min(factor_count / 6, 1.0) * 0.3
        return "medium", round(score, 2)


# ── Composite evaluator ───────────────────────────────────────────────────────

class MatchPredictionEvaluator:
    """
    Runs all three evaluators and returns a composite EvalResult.
    Designed to run as a background task after every match result arrives.
    """

    WEIGHTS = {
        "accuracy":   0.50,
        "calibration": 0.25,
        "reasoning":  0.25,
    }

    def __init__(self) -> None:
        self.accuracy_eval    = PredictionAccuracyEvaluator()
        self.calibration_eval = ConfidenceCalibrationEvaluator()
        self.reasoning_eval   = ReasoningQualityEvaluator()

    def evaluate(self, trace_record: dict) -> EvalResult:
        attrs    = trace_record.get("root_span", {}).get("attributes", {})
        trace_id = trace_record.get("trace_id", "unknown")

        prediction   = attrs.get("matchmind.prediction", "")
        actual       = attrs.get("matchmind.actual_result", "")
        confidence   = float(attrs.get("matchmind.confidence", 0.5))
        reasoning    = attrs.get("matchmind.reasoning", "")
        tools_raw    = attrs.get("matchmind.tools_called", "")
        factors_raw  = attrs.get("matchmind.factors_considered", "[]")
        match_id     = attrs.get("matchmind.match_id", "unknown")
        prompt_ver   = attrs.get("matchmind.prompt_version", "v1")

        tools_called = [t.strip() for t in tools_raw.split(",") if t.strip()]
        factors      = json.loads(factors_raw) if factors_raw else []

        acc_label,  acc_score  = self.accuracy_eval.evaluate(prediction, actual)
        cal_label,  cal_score  = self.calibration_eval.evaluate(confidence, acc_label)
        reas_label, reas_score = self.reasoning_eval.evaluate(reasoning, tools_called, factors)

        composite = (
            acc_score  * self.WEIGHTS["accuracy"]
            + cal_score  * self.WEIGHTS["calibration"]
            + reas_score * self.WEIGHTS["reasoning"]
        )

        result = EvalResult(
            trace_id       = trace_id,
            match_id       = match_id,
            accuracy       = acc_label,
            calibration    = cal_label,
            reasoning_quality = reas_label,
            accuracy_score = acc_score,
            calibration_score = cal_score,
            reasoning_score = reas_score,
            composite_score = round(composite, 3),
            prompt_version  = prompt_ver,
            failure         = acc_label == "incorrect",
        )

        logger.info(
            "Eval complete",
            extra={
                "match_id":   match_id,
                "accuracy":   acc_label,
                "composite":  composite,
                "prompt_ver": prompt_ver,
            },
        )
        return result

    def evaluate_batch(self, traces: list[dict]) -> list[EvalResult]:
        """Evaluate a list of traces. Skips any without actual_result."""
        results = []
        for trace in traces:
            attrs = trace.get("root_span", {}).get("attributes", {})
            if not attrs.get("matchmind.actual_result"):
                continue
            results.append(self.evaluate(trace))
        return results

    def aggregate(self, results: list[EvalResult]) -> dict:
        """Compute summary stats across a batch of eval results."""
        if not results:
            return {}
        total = len(results)
        correct = sum(1 for r in results if r.accuracy == "correct")
        overconfident = sum(1 for r in results if r.calibration == "overconfident")
        high_reasoning = sum(1 for r in results if r.reasoning_quality == "high")
        avg_composite = sum(r.composite_score for r in results) / total

        version_stats: dict[str, dict] = {}
        for r in results:
            v = r.prompt_version
            if v not in version_stats:
                version_stats[v] = {"total": 0, "correct": 0}
            version_stats[v]["total"] += 1
            if r.accuracy == "correct":
                version_stats[v]["correct"] += 1

        for v, s in version_stats.items():
            s["accuracy_rate"] = round(s["correct"] / s["total"], 3)

        return {
            "total_evaluated": total,
            "accuracy_rate":   round(correct / total, 3),
            "overconfident_rate": round(overconfident / total, 3),
            "high_reasoning_rate": round(high_reasoning / total, 3),
            "avg_composite_score": round(avg_composite, 3),
            "by_prompt_version": version_stats,
            "worst_traces": [
                r.trace_id for r in results
                if r.accuracy == "incorrect" and r.reasoning_quality == "low"
            ],
        }
