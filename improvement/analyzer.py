"""
Trace Failure Analyzer.

June 2026 fix: failures are now read from the durable PredictionStore —
the same records the evaluators write — instead of from Phoenix REST
queries against endpoints that don't exist. Phoenix is used (best-effort,
via PhoenixSync) only to persist new prompt versions for durability and
experiment tracking. The loop no longer silently no-ops when Phoenix is
unreachable.
"""
import logging
from typing import Any, Optional

from agent.prediction_store import store
from agent.prompts.templates import (
    get_active_prediction_prompt,
    get_active_version,
)

logger = logging.getLogger("matchmind.improvement.analyzer")


class TraceFailureAnalyzer:

    def __init__(self, phoenix_sync: Optional[Any] = None) -> None:
        """
        Args:
            phoenix_sync: Optional PhoenixSync instance for best-effort
                          prompt persistence. The analyzer works fully
                          without it.
        """
        self._sync = phoenix_sync

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_recent_failures(
        self,
        project_name: str = "",
        limit: int = 20,
    ) -> list[dict]:
        """
        Return structured failure records (evaluated + incorrect), newest first.
        Source of truth: PredictionStore.
        """
        rows = store.get_failures(limit=limit)
        failures = []
        for rec in rows:
            failures.append({
                "trace_id":          rec.get("trace_id"),
                "span_id":           rec.get("span_id"),
                "match_id":          rec.get("match_id"),
                "home_team":         rec.get("home_team"),
                "away_team":         rec.get("away_team"),
                "prediction":        rec.get("prediction"),
                "actual_result":     rec.get("actual_result"),
                "confidence":        float(rec.get("confidence", 0.5)),
                "reasoning":         rec.get("reasoning", ""),
                "factors":           rec.get("factors", []),
                "tools_called":      rec.get("tools_called", []),
                "tool_count":        int(rec.get("tool_count", 0)),
                "prompt_version":    rec.get("prompt_version", "unknown"),
                "calibration":       rec.get("calibration"),
                "reasoning_quality": rec.get("reasoning_quality"),
            })
        logger.info("Failures retrieved from store", extra={"count": len(failures)})
        return failures

    async def extract_failure_patterns(self, failures: list[dict]) -> dict:
        """
        Aggregate failure records into actionable pattern statistics.
        Returns a dict the improvement loop and Gemini can reason over.
        """
        if not failures:
            return {}

        total = len(failures)
        patterns: dict[str, Any] = {
            "total_failures": total,
            "overconfident_count": 0,
            "underconfident_count": 0,
            "missing_injury_check_count": 0,
            "low_tool_usage_count": 0,
            "no_h2h_check_count": 0,
            "shallow_reasoning_count": 0,
            "by_prompt_version": {},
            "wrong_high_confidence_count": 0,
        }

        for f in failures:
            pv = f.get("prompt_version", "unknown")
            patterns["by_prompt_version"].setdefault(pv, 0)
            patterns["by_prompt_version"][pv] += 1

            if f.get("calibration") == "overconfident":
                patterns["overconfident_count"] += 1
            if f.get("calibration") == "underconfident":
                patterns["underconfident_count"] += 1

            tools = [str(t).lower() for t in f.get("tools_called", [])]
            if not any("injur" in t for t in tools):
                patterns["missing_injury_check_count"] += 1
            if not any("head" in t or "h2h" in t for t in tools):
                patterns["no_h2h_check_count"] += 1
            if f.get("tool_count", 0) < 4:
                patterns["low_tool_usage_count"] += 1

            if f.get("reasoning_quality") == "low":
                patterns["shallow_reasoning_count"] += 1

            if f.get("confidence", 0) >= 0.70:
                patterns["wrong_high_confidence_count"] += 1

        patterns["overconfident_rate"] = round(patterns["overconfident_count"] / total, 2)
        patterns["injury_miss_rate"] = round(patterns["missing_injury_check_count"] / total, 2)
        patterns["shallow_reasoning_rate"] = round(patterns["shallow_reasoning_count"] / total, 2)

        return patterns

    async def get_current_prompt(self, project_name: str = "") -> dict:
        """Active prompt from the local registry (durable via store)."""
        return {
            "version": get_active_version(),
            "content": get_active_prediction_prompt(),
            "id": None,
        }

    async def create_improved_prompt(
        self,
        project_name: str,
        content: str,
        version_tag: str,
        description: str,
    ) -> dict:
        """
        Persist a new prompt version. Local registration is handled by the
        loop (register_new_version); this pushes to Phoenix best-effort.
        """
        prompt_id = None
        if self._sync and getattr(self._sync, "enabled", False):
            from agent.config import config
            prompt_id = await self._sync.push_prompt_version(
                content=content,
                description=f"{version_tag}: {description}",
                model_name=config.GEMINI_MODEL,
            )
        return {"id": prompt_id, "version": version_tag}

    async def get_performance_by_version(self, project_name: str = "") -> dict:
        """Accuracy stats per prompt version from the store."""
        return store.accuracy_by_version()
