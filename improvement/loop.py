"""
Self-Improvement Loop — the core innovation of MatchMind.

Triggered automatically after each match result is submitted.

The cycle (June 2026 revision):
  0. GUARD      — Regression check: if the active prompt version is
                  measurably WORSE than its predecessor, roll back first.
                  (Previously `min_accuracy_delta` was accepted in the
                  constructor and never used — new prompts activated
                  unconditionally with no rollback path.)
  1. FETCH      — Pull recent failures from the durable PredictionStore
  2. EVALUATE   — Aggregate the already-computed eval results
  3. ANALYSE    — Extract failure patterns
  4. GENERATE   — Use Gemini to write an improved prompt
  5. PERSIST    — Register locally (live via instruction provider) and
                  push to Phoenix Prompt Management (best-effort)
  6. REPORT     — Return structured improvement report

The agent watches itself fail and rewrites its own instructions —
with a guardrail now, so a bad rewrite can't become permanent.
"""
import json
import logging
from datetime import datetime, timezone
from opentelemetry import trace

from agent.config import config
from agent.prediction_store import store
from observability.evaluators import MatchPredictionEvaluator
from improvement.analyzer import TraceFailureAnalyzer
from agent.prompts.templates import (
    register_new_version,
    rollback_active_version,
    get_active_version,
)

tracer = trace.get_tracer("matchmind.improvement_loop")
logger = logging.getLogger("matchmind.improvement_loop")

# Minimum evaluated predictions a version needs before regression
# comparison is considered statistically meaningful (small, demo-scale).
MIN_EVALS_FOR_REGRESSION_CHECK = 3


class SelfImprovementLoop:

    def __init__(
        self,
        gemini_client,
        analyzer: TraceFailureAnalyzer,
        evaluator: MatchPredictionEvaluator,
        project_name: str,
        min_failures_to_trigger: int = 5,
        min_accuracy_delta: float = 0.05,
    ) -> None:
        self.gemini    = gemini_client
        self.analyzer  = analyzer
        self.evaluator = evaluator
        self.project   = project_name
        self.min_failures = min_failures_to_trigger
        self.min_delta    = min_accuracy_delta

    @property
    def _cycle_count(self) -> int:
        return store.cycle_count

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, trigger_match_ids: list[str] | None = None) -> dict:
        """Execute one full improvement cycle. Returns an improvement report."""
        with tracer.start_as_current_span("improvement_loop.cycle") as span:
            ts = datetime.now(timezone.utc).isoformat()
            span.set_attribute("loop.trigger_matches", json.dumps(trigger_match_ids or []))
            span.set_attribute("loop.timestamp", ts)

            logger.info("=== Improvement loop started ===")

            # ── Step 0: Regression guard ──────────────────────────────────────
            rollback = self.check_regression()
            if rollback:
                span.set_attribute("loop.rollback", rollback["from_version"])
                logger.warning("Regression detected — rolled back %s -> %s",
                               rollback["from_version"], rollback["to_version"])

            # ── Step 1: Fetch recent failures (from durable store) ────────────
            logger.info("Step 1/6 — Fetching failures from store")
            failures = await self.analyzer.get_recent_failures(
                project_name=self.project, limit=30,
            )

            if len(failures) < self.min_failures:
                msg = f"Only {len(failures)} failures found (min: {self.min_failures}). Skipping."
                logger.info(msg)
                span.set_attribute("loop.result", "insufficient_data")
                report = {"status": "skipped", "reason": msg,
                          "failures_found": len(failures)}
                if rollback:
                    report["rollback"] = rollback
                return report

            # ── Step 2: Aggregate eval results ────────────────────────────────
            logger.info("Step 2/6 — Evaluating %d failure records", len(failures))
            eval_results = self.evaluator.evaluate_batch(
                [{
                    "trace_id": f["trace_id"],
                    "root_span": {"attributes": {
                        "matchmind.prediction":         f["prediction"],
                        "matchmind.actual_result":      f["actual_result"],
                        "matchmind.confidence":         f["confidence"],
                        "matchmind.reasoning":          f["reasoning"],
                        "matchmind.factors_considered": json.dumps(f["factors"]),
                        "matchmind.tools_called":       ",".join(f["tools_called"]),
                        "matchmind.match_id":           f["match_id"],
                        "matchmind.prompt_version":     f["prompt_version"],
                    }},
                } for f in failures]
            )
            summary = self.evaluator.aggregate(eval_results)

            # ── Step 3: Extract patterns ──────────────────────────────────────
            logger.info("Step 3/6 — Extracting failure patterns")
            patterns = await self.analyzer.extract_failure_patterns(failures)
            span.set_attribute("loop.patterns", json.dumps(
                {k: v for k, v in patterns.items() if isinstance(v, (int, float))}
            ))

            # ── Step 4: Generate improved prompt ──────────────────────────────
            logger.info("Step 4/6 — Generating improved prompt with Gemini")
            current_prompt = await self.analyzer.get_current_prompt(self.project)
            new_prompt = await self._generate_improved_prompt(
                current_prompt=current_prompt["content"],
                patterns=patterns,
                examples=failures[:5],
                summary=summary,
            )

            if not new_prompt or len(new_prompt.strip()) < 200:
                span.set_attribute("loop.result", "generation_failed")
                return {"status": "skipped",
                        "reason": "Generated prompt was empty or implausibly short",
                        "failures_found": len(failures)}

            # ── Step 5: Persist (local = live; Phoenix = durable) ─────────────
            logger.info("Step 5/6 — Persisting new prompt version")
            version_tag = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            description = self._build_description(patterns, summary)

            created = await self.analyzer.create_improved_prompt(
                project_name=self.project,
                content=new_prompt,
                version_tag=version_tag,
                description=description,
            )

            # Goes live on the next agent run via the instruction provider.
            register_new_version(version_tag, new_prompt)
            cycle = store.increment_cycle()

            # ── Step 6: Report ────────────────────────────────────────────────
            report = {
                "status":            "improved",
                "cycle":             cycle,
                "timestamp":         ts,
                "failures_analysed": len(failures),
                "eval_summary":      summary,
                "patterns":          {k: v for k, v in patterns.items()
                                      if isinstance(v, (int, float))},
                "new_version":       version_tag,
                "previous_version":  current_prompt["version"],
                "description":       description,
                "prompt_id":         created.get("id"),
            }
            if rollback:
                report["rollback"] = rollback

            span.set_attribute("loop.result", "improved")
            span.set_attribute("loop.new_version", version_tag)
            span.set_attribute("loop.failures", len(failures))

            logger.info("=== Improvement loop complete: %s ===", version_tag)
            return report

    # ── Regression guard ──────────────────────────────────────────────────────

    def check_regression(self) -> dict | None:
        """
        Compare the active prompt version's accuracy against its predecessor.
        If the active version has enough evaluated predictions and is worse
        by more than min_accuracy_delta, roll back.

        Returns a rollback report dict, or None if no action taken.
        """
        active = get_active_version()
        if active == "v1":
            return None  # baseline can't be rolled back

        stats = store.accuracy_by_version()
        active_stats = stats.get(active)
        if not active_stats or active_stats["evaluated"] < MIN_EVALS_FOR_REGRESSION_CHECK:
            return None

        # Compare against the nearest PREDECESSOR with actual evidence —
        # the immediate previous version may have been replaced before it
        # accumulated any evaluated predictions.
        order = [v for v in store._data["version_order"]
                 if v not in store._data["rolled_back"]]
        candidates = list(reversed(order[:order.index(active)])) + ["v1"] \
            if active in order else ["v1"]
        prev, prev_stats = None, None
        for cand in candidates:
            s = stats.get(cand)
            if s and s["evaluated"]:
                prev, prev_stats = cand, s
                break
        if not prev_stats:
            return None

        active_acc = active_stats["accuracy_rate"] or 0.0
        prev_acc = prev_stats["accuracy_rate"] or 0.0

        if active_acc + self.min_delta < prev_acc:
            restored = rollback_active_version()
            return {
                "from_version": active,
                "to_version": restored,
                "active_accuracy": active_acc,
                "previous_accuracy": prev_acc,
                "delta_threshold": self.min_delta,
            }
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_improved_prompt(
        self,
        current_prompt: str,
        patterns: dict,
        examples: list[dict],
        summary: dict,
    ) -> str:
        """Use Gemini to write a better system prompt based on failure analysis."""
        example_text = "\n".join(
            f"  - {e['home_team']} vs {e['away_team']}: "
            f"predicted {e['prediction']}, "
            f"actual {e['actual_result']}, "
            f"confidence {e['confidence']:.2f}, "
            f"tools used: {e['tool_count']}"
            for e in examples
        )

        meta_prompt = f"""You are improving a World Cup prediction AI agent's system prompt.

CURRENT PROMPT (active version):
---
{current_prompt or '[not available — use best judgment]'}
---

FAILURE ANALYSIS (last {patterns.get('total_failures', 0)} incorrect predictions):
  - Overconfident predictions:         {patterns.get('overconfident_count', 0)} ({patterns.get('overconfident_rate', 0):.0%})
  - Missing injury check:              {patterns.get('missing_injury_check_count', 0)} ({patterns.get('injury_miss_rate', 0):.0%})
  - Low tool usage (< 4 tools):        {patterns.get('low_tool_usage_count', 0)}
  - Missing head-to-head check:        {patterns.get('no_h2h_check_count', 0)}
  - Shallow reasoning (low quality):   {patterns.get('shallow_reasoning_count', 0)} ({patterns.get('shallow_reasoning_rate', 0):.0%})
  - High-confidence wrong predictions: {patterns.get('wrong_high_confidence_count', 0)}

EXAMPLE FAILURES:
{example_text}

ACCURACY BY PROMPT VERSION:
{json.dumps(summary.get('by_prompt_version', {}), indent=2)}

INSTRUCTIONS:
Write an IMPROVED system prompt that directly addresses the above failure patterns.

Specifically:
1. If injury_miss_rate > 0.30: add a HARD STOP requiring injury check before any analysis
2. If overconfident_rate > 0.25: strengthen the confidence calibration rules
3. If low tool usage > 30%: add explicit consequences for skipping data tools
4. If shallow reasoning > 0.30: require reasoning word count > 80 words minimum
5. Maintain all existing structure and format requirements, including the
   mandatory store_prediction() call — but instruct the agent to set
   prompt_version to the CURRENT active version it is running.

Return ONLY the improved system prompt text.
Do NOT include any preamble, explanation, or markdown fences.
"""

        response = await self.gemini.aio.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=meta_prompt,
        )
        return (response.text or "").strip()

    def _build_description(self, patterns: dict, summary: dict) -> str:
        """Concise human-readable description of what changed and why."""
        reasons = []
        if patterns.get("overconfident_rate", 0) > 0.25:
            reasons.append("strengthened confidence calibration")
        if patterns.get("injury_miss_rate", 0) > 0.30:
            reasons.append("enforced mandatory injury check")
        if patterns.get("low_tool_usage_count", 0) > 5:
            reasons.append("raised minimum tool usage to 4")
        if patterns.get("shallow_reasoning_rate", 0) > 0.30:
            reasons.append("added reasoning depth requirement")
        if not reasons:
            reasons.append("general quality improvement")

        acc = summary.get("accuracy_rate", 0)
        return (
            f"Auto-generated improvement (accuracy baseline: {acc:.0%}). "
            f"Changes: {'; '.join(reasons)}."
        )
