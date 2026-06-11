"""
Trace Failure Analyzer.

Uses Phoenix MCP tools (injected via ADK at runtime) to:
  1. Query the agent's own worst-performing traces
  2. Extract structured failure patterns
  3. Pull the current active prompt version

This is the "eyes" of the self-improvement loop — it reads
what Phoenix knows about us and turns it into actionable intelligence.
"""
import json
import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger("matchmind.improvement.analyzer")

# Type alias for a Phoenix MCP tool callable
PhoenixTool = Callable[..., Awaitable[dict[str, Any]]]


class TraceFailureAnalyzer:

    def __init__(self, phoenix_tools: dict[str, PhoenixTool]) -> None:
        """
        Args:
            phoenix_tools: Dict of tool_name → async callable
                           sourced from the Phoenix MCPToolset at runtime.
                           Expected keys: get_traces, get_spans,
                           get_prompts, create_prompt, get_experiments.
        """
        self._t = phoenix_tools

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_recent_failures(
        self,
        project_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Query Phoenix for traces where accuracy eval = incorrect.
        Returns a list of structured failure records.
        """
        logger.info("Querying Phoenix for recent failures", extra={"limit": limit})

        raw = await self._t["get_traces"](
            project_name=project_name,
            filter_condition="evals['accuracy'].label == 'incorrect'",
            limit=limit,
            sort_by="start_time",
            sort_direction="desc",
        )

        failures = []
        for trace in raw.get("data", []):
            attrs = trace.get("root_span", {}).get("attributes", {})
            failures.append({
                "trace_id":          trace.get("trace_id"),
                "match_id":          attrs.get("matchmind.match_id"),
                "home_team":         attrs.get("matchmind.home_team"),
                "away_team":         attrs.get("matchmind.away_team"),
                "prediction":        attrs.get("matchmind.prediction"),
                "actual_result":     attrs.get("matchmind.actual_result"),
                "confidence":        float(attrs.get("matchmind.confidence", 0.5)),
                "reasoning":         attrs.get("matchmind.reasoning", ""),
                "factors":           json.loads(attrs.get("matchmind.factors_considered", "[]")),
                "tools_called":      attrs.get("matchmind.tools_called", "").split(","),
                "tool_count":        int(attrs.get("matchmind.tool_count", 0)),
                "prompt_version":    attrs.get("matchmind.prompt_version", "unknown"),
                "calibration":       attrs.get("eval.calibration"),
                "reasoning_quality": attrs.get("eval.reasoning_quality"),
            })

        logger.info("Failures retrieved", extra={"count": len(failures)})
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
            # Calibration issues
            "overconfident_count": 0,
            "underconfident_count": 0,
            # Data coverage issues
            "missing_injury_check_count": 0,     # injury tool not called
            "low_tool_usage_count": 0,            # < 4 tools used
            "no_h2h_check_count": 0,              # head-to-head not called
            # Reasoning depth
            "shallow_reasoning_count": 0,         # reasoning_quality == low
            # By prompt version (to track regressions)
            "by_prompt_version": {},
            # Confidence distribution of wrong predictions
            "wrong_high_confidence_count": 0,     # confidence >= 0.70 AND incorrect
        }

        for f in failures:
            pv = f.get("prompt_version", "unknown")
            if pv not in patterns["by_prompt_version"]:
                patterns["by_prompt_version"][pv] = 0
            patterns["by_prompt_version"][pv] += 1

            if f.get("calibration") == "overconfident":
                patterns["overconfident_count"] += 1
            if f.get("calibration") == "underconfident":
                patterns["underconfident_count"] += 1

            tools = [t.lower() for t in f.get("tools_called", [])]
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

        # Compute rates
        patterns["overconfident_rate"] = round(
            patterns["overconfident_count"] / total, 2
        )
        patterns["injury_miss_rate"] = round(
            patterns["missing_injury_check_count"] / total, 2
        )
        patterns["shallow_reasoning_rate"] = round(
            patterns["shallow_reasoning_count"] / total, 2
        )

        return patterns

    async def get_current_prompt(self, project_name: str) -> dict:
        """
        Retrieve the active prompt version from Phoenix Prompt Management.
        Falls back to {"version": "v1", "content": "", "id": None} if empty.
        """
        response = await self._t["get_prompts"](project_name=project_name)
        prompts  = response.get("data", [])

        if not prompts:
            return {"version": "v1", "content": "", "id": None}

        # Return highest version number
        latest = sorted(
            prompts,
            key=lambda p: p.get("version_num", 0),
            reverse=True,
        )[0]
        return {
            "version": latest.get("version", "v1"),
            "content": latest.get("content", ""),
            "id":      latest.get("id"),
        }

    async def create_improved_prompt(
        self,
        project_name: str,
        content: str,
        version_tag: str,
        description: str,
    ) -> dict:
        """
        Write a new prompt version to Phoenix Prompt Management.
        This makes the new version available for experiment tracking
        and allows the agent to fetch it at next startup.
        """
        result = await self._t["create_prompt"](
            project_name=project_name,
            name="match_prediction_prompt",
            version=version_tag,
            content=content,
            description=description,
            tags=["auto_generated", "improvement_loop"],
        )
        logger.info(
            "New prompt version created in Phoenix",
            extra={"version": version_tag, "description": description},
        )
        return result

    async def get_performance_by_version(self, project_name: str) -> list[dict]:
        """
        Retrieve experiment results per prompt version from Phoenix.
        Used by the dashboard performance endpoint.
        """
        response = await self._t["get_experiments"](
            project_name=project_name,
            dataset_name="match_predictions",
        )
        return response.get("data", [])
