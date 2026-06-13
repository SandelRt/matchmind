"""
MatchMind logic tests — no ADK / network required.
Run:  python tests/run_tests.py
"""
import asyncio
import json
import os
import sys
import tempfile

# Isolated store BEFORE any agent imports
_tmp = tempfile.mkdtemp()
os.environ["STORE_PATH"] = os.path.join(_tmp, "store.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f" FAIL {name}")


# ── 1. Store roundtrip ────────────────────────────────────────────────────────
print("\n[1] PredictionStore")
from agent.prediction_store import store

store.record_prediction("M1", {
    "trace_id": "t1", "span_id": "s1", "match_id": "M1",
    "home_team": "Brazil", "away_team": "France",
    "prediction": "2-1", "confidence": 0.8, "reasoning": "Strong form " * 12,
    "factors": ["form", "injuries", "h2h", "standings"],
    "tools_called": ["get_team_form", "get_team_injuries", "get_head_to_head", "get_tournament_standings"],
    "tool_count": 4, "prompt_version": "v1",
})
check("record + get", store.get("M1")["prediction"] == "2-1")
rec = store.record_result("M1", "0-2", "away_win")
check("record_result returns record", rec and rec["actual_result"] == "0-2")
check("unknown match_id -> None", store.record_result("NOPE", "1-0", "home_win") is None)

# ── 2. Eval pipeline (process_match_result path, no Phoenix) ─────────────────
print("\n[2] Eval write-back via process_match_result")
from agent.tools.prediction import process_match_result

res = asyncio.run(process_match_result("M1", 0, 2))
check("evaluated incorrect", res["accuracy"] == "incorrect")
check("eval persisted to store", store.get("M1")["evaluated"] is True)
check("idempotent re-eval", asyncio.run(process_match_result("M1", 0, 2))["status"] == "already_evaluated")
check("no stored prediction handled", asyncio.run(process_match_result("M404", 1, 0))["status"] == "no_stored_prediction")

# correct prediction case
store.record_prediction("M2", {
    "trace_id": "t2", "span_id": "s2", "match_id": "M2",
    "home_team": "Spain", "away_team": "Croatia",
    "prediction": "2-1", "confidence": 0.65, "reasoning": "Balanced sides " * 12,
    "factors": ["form", "injuries", "h2h", "standings"],
    "tools_called": ["get_team_form", "get_team_injuries", "get_head_to_head", "get_tournament_standings"],
    "tool_count": 4, "prompt_version": "v1",
})
res2 = asyncio.run(process_match_result("M2", 3, 1))
check("correct direction scored correct", res2["accuracy"] == "correct")
check("failures list has only M1", [f["match_id"] for f in store.get_failures()] == ["M1"])

# ── 2b. Gemini schema fix: tool accepts STRING args (origin 7790500) ─────────
print("\n[2b] String-param tool call (Gemini 400 fix)")
from agent.tools.prediction import store_prediction, _to_list

res_s = asyncio.run(store_prediction(
    match_id="M3", home_team="Mexico", away_team="Ghana",
    predicted_score="1-1", result_direction="draw", confidence=0.55,
    reasoning="Evenly matched " * 15,
    factors_considered="home form strong; injuries minimal; h2h split; both qualified",
    uncertainty_factors="weather; rotation risk",
    tools_called="get_team_form, get_team_injuries, get_head_to_head, get_tournament_standings",
    prompt_version="v1",
))
m3 = store.get("M3")
check("string args stored", res_s["status"] == "stored")
check("factors normalized to list", m3["factors"] == [
    "home form strong", "injuries minimal", "h2h split", "both qualified"])
check("tools normalized to list", m3["tool_count"] == 4)
check("json-array string accepted", _to_list('["a","b"]') == ["a", "b"])
check("real list still accepted", _to_list(["x", "y"]) == ["x", "y"])
res_s2 = asyncio.run(process_match_result("M3", 1, 1))
check("string-arg prediction evaluates", res_s2["accuracy"] == "correct")

# ── 3. Analyzer patterns ──────────────────────────────────────────────────────
print("\n[3] TraceFailureAnalyzer")
from improvement.analyzer import TraceFailureAnalyzer

analyzer = TraceFailureAnalyzer(phoenix_sync=None)
failures = asyncio.run(analyzer.get_recent_failures(limit=10))
check("failures from store", len(failures) == 1 and failures[0]["match_id"] == "M1")
patterns = asyncio.run(analyzer.extract_failure_patterns(failures))
check("patterns computed", patterns["total_failures"] == 1)
check("high-confidence wrong counted", patterns["wrong_high_confidence_count"] == 1)
check("injury check detected (no miss)", patterns["missing_injury_check_count"] == 0)

# ── 4. Templates: provider + registry ─────────────────────────────────────────
print("\n[4] Prompt registry + instruction provider")
from agent.prompts.templates import (
    get_active_prediction_prompt, get_active_version,
    register_new_version, PROMPT_V1,
)

check("baseline active", get_active_version() == "v1")
check("baseline content", get_active_prediction_prompt() == PROMPT_V1)
register_new_version("v_test_1", "IMPROVED PROMPT ONE " * 20)
check("new version active immediately", get_active_version() == "v_test_1")
check("provider returns new content", get_active_prediction_prompt().startswith("IMPROVED PROMPT ONE"))

# ── 5. Improvement loop end-to-end with fake Gemini ──────────────────────────
print("\n[5] SelfImprovementLoop")
from improvement.loop import SelfImprovementLoop
from observability.evaluators import MatchPredictionEvaluator

class FakeAio:
    class models:
        @staticmethod
        async def generate_content(model, contents):
            class R: text = "REWRITTEN SYSTEM PROMPT. " * 30
            return R()
class FakeGemini:
    aio = FakeAio()

loop = SelfImprovementLoop(
    gemini_client=FakeGemini(),
    analyzer=analyzer,
    evaluator=MatchPredictionEvaluator(),
    project_name="test",
    min_failures_to_trigger=1,
    min_accuracy_delta=0.05,
)
report = asyncio.run(loop.run(trigger_match_ids=["M1"]))
check("loop status improved", report["status"] == "improved")
check("new version registered + live",
      get_active_prediction_prompt().startswith("REWRITTEN SYSTEM PROMPT"))
check("cycle count persisted", store.cycle_count == 1)
check("previous version recorded", report["previous_version"] == "v_test_1")

# insufficient-data path
loop2 = SelfImprovementLoop(FakeGemini(), analyzer, MatchPredictionEvaluator(),
                            "test", min_failures_to_trigger=99)
check("skips when insufficient failures",
      asyncio.run(loop2.run())["status"] == "skipped")

# short-output guard
class EmptyAio:
    class models:
        @staticmethod
        async def generate_content(model, contents):
            class R: text = "too short"
            return R()
class EmptyGemini:
    aio = EmptyAio()
loop3 = SelfImprovementLoop(EmptyGemini(), analyzer, MatchPredictionEvaluator(),
                            "test", min_failures_to_trigger=1)
check("rejects implausibly short prompt",
      asyncio.run(loop3.run())["status"] == "skipped")

# ── 6. Regression guard + rollback ────────────────────────────────────────────
print("\n[6] Regression rollback")
active = get_active_version()  # the REWRITTEN version
# 3 evaluated failures under the active version vs M2 correct under v1
for i, mid in enumerate(["R1", "R2", "R3"]):
    store.record_prediction(mid, {
        "trace_id": f"tr{i}", "span_id": f"sr{i}", "match_id": mid,
        "home_team": "A", "away_team": "B",
        "prediction": "1-0", "confidence": 0.8, "reasoning": "x " * 90,
        "factors": ["form"], "tools_called": ["get_team_form"],
        "tool_count": 1, "prompt_version": active,
    })
    store.record_result(mid, "0-1", "away_win")
    store.attach_eval(mid, {"accuracy": "incorrect", "accuracy_score": 0.0,
                            "calibration": "overconfident", "calibration_score": 0.0,
                            "reasoning_quality": "low", "reasoning_score": 0.0,
                            "composite_score": 0.0})
rb = loop.check_regression()
check("regression detected", rb is not None and rb["from_version"] == active)
check("rolled back off bad version", get_active_version() != active)

# ── 7. Store survives reload (restart simulation) ─────────────────────────────
print("\n[7] Durability")
from agent.prediction_store import PredictionStore
fresh = PredictionStore(path=os.environ["STORE_PATH"])
check("predictions survive reload", fresh.get("M1") is not None)
check("cycle count survives reload", fresh.cycle_count == 1)
check("rolled-back version excluded after reload",
      fresh._data["active_version"] != active)

print(f"\n{'='*40}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
