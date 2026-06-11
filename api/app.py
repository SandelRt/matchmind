"""
MatchMind FastAPI server — deployed to Google Cloud Run.

Endpoints:
  POST /predict          Generate a match prediction
  POST /results          Submit actual match result (triggers improvement loop)
  POST /chat             Multi-turn conversation
  POST /improve          Manually trigger improvement cycle
  GET  /performance      Agent performance metrics
  GET  /health           Health check
  GET  /                 Dashboard (serves frontend/index.html)
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import google.genai as genai

from agent.agent import build_agent
from agent.config import config
from observability.tracing import setup_tracing, shutdown_tracing
from observability.evaluators import MatchPredictionEvaluator
from observability.phoenix_client import build_phoenix_tools
from improvement.analyzer import TraceFailureAnalyzer
from improvement.loop import SelfImprovementLoop

logger = logging.getLogger("matchmind.api")

# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("MatchMind starting up...")

    # 1. Wire Phoenix tracing FIRST (before any agent activity)
    setup_tracing(
        phoenix_api_key=config.PHOENIX_API_KEY,
        phoenix_base_url=config.PHOENIX_BASE_URL,
        project_name=config.PHOENIX_PROJECT_NAME,
    )
    logger.info("Phoenix tracing active → %s", config.PHOENIX_BASE_URL)

    # 2. Build the ADK agent
    _state["agent"] = build_agent()
    logger.info("ADK agent initialized (model: %s)", config.GEMINI_MODEL)

    # 3. Build improvement loop dependencies
    gemini = genai.Client()
    evaluator = MatchPredictionEvaluator()

    # Build Phoenix REST tool callables for the improvement loop.
    # These call Phoenix directly over HTTPS — separate from the MCP server
    # the ADK agent uses for in-conversation self-introspection.
    phoenix_tools = build_phoenix_tools(
        api_key=config.PHOENIX_API_KEY,
        base_url=config.PHOENIX_BASE_URL,
        project_name=config.PHOENIX_PROJECT_NAME,
    )
    analyzer = TraceFailureAnalyzer(phoenix_tools=phoenix_tools)

    improvement_loop = SelfImprovementLoop(
        gemini_client=gemini,
        analyzer=analyzer,
        evaluator=evaluator,
        project_name=config.PHOENIX_PROJECT_NAME,
        min_failures_to_trigger=config.IMPROVEMENT_TRIGGER_THRESHOLD,
        min_accuracy_delta=config.IMPROVEMENT_MIN_DELTA,
    )

    _state["evaluator"]        = evaluator
    _state["gemini"]           = gemini
    _state["analyzer"]         = analyzer
    _state["improvement_loop"] = improvement_loop

    logger.info("✅ MatchMind ready — improvement loop wired")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    shutdown_tracing()
    logger.info("MatchMind shut down cleanly")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MatchMind",
    description=(
        "Self-improving World Cup 2026 prediction agent. "
        "Powered by Gemini + Google ADK + Arize Phoenix."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve frontend dashboard
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    match_id:   str = Field(..., example="WC26_R16_01")
    home_team:  str = Field(..., example="Brazil")
    away_team:  str = Field(..., example="France")
    match_date: str = Field(..., example="2026-06-28T20:00:00Z")
    stage:      str = Field("group", example="round_of_16")


class ResultRequest(BaseModel):
    match_id:    str = Field(..., example="WC26_R16_01")
    home_goals:  int = Field(..., ge=0, example=2)
    away_goals:  int = Field(..., ge=0, example=1)


class ChatRequest(BaseModel):
    message:    str = Field(..., example="How will Argentina do in the quarter finals?")
    session_id: str = Field("default", example="user_123")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    index = _frontend / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "MatchMind API", "docs": "/docs"})


@app.post("/predict")
async def predict_match(req: PredictionRequest):
    """
    Generate a match prediction.

    The agent calls WC26 MCP tools to gather live data, reasons over it,
    stores a richly attributed trace span in Phoenix, and returns
    a structured prediction.
    """
    agent = _state.get("agent")
    if not agent:
        raise HTTPException(503, "Agent not ready")

    prompt = (
        f"Predict the World Cup match: {req.home_team} vs {req.away_team} "
        f"on {req.match_date} (stage: {req.stage}, match_id: {req.match_id}). "
        f"Follow the mandatory 6-step prediction protocol. "
        f"Store the prediction with match_id '{req.match_id}'."
    )

    try:
        result = await agent.run_async(prompt)
        return {
            "status":   "ok",
            "match_id": req.match_id,
            "result":   result,
        }
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/results")
async def submit_result(req: ResultRequest, background: BackgroundTasks):
    """
    Submit the actual match result.

    This:
    1. Updates the stored prediction trace with the actual score
    2. Runs the evaluation pipeline
    3. Triggers the self-improvement loop in the background
    """
    agent = _state.get("agent")
    if not agent:
        raise HTTPException(503, "Agent not ready")

    actual_score = f"{req.home_goals}-{req.away_goals}"
    update_prompt = (
        f"The match with ID '{req.match_id}' has ended. "
        f"Final score: {actual_score} (home {req.home_goals}, away {req.away_goals}). "
        f"Call update_prediction_with_result() to record this result."
    )

    await agent.run_async(update_prompt)
    background.add_task(_run_improvement_loop, [req.match_id])

    return {
        "status":                  "result_recorded",
        "match_id":                req.match_id,
        "actual_score":            actual_score,
        "improvement_loop_queued": True,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Open-ended conversation with MatchMind.

    Ask about upcoming matches, team form, past predictions,
    or let the agent self-report on its performance history via Phoenix MCP.
    """
    agent = _state.get("agent")
    if not agent:
        raise HTTPException(503, "Agent not ready")

    try:
        result = await agent.run_async(req.message, session_id=req.session_id)
        return {"response": result, "session_id": req.session_id}
    except Exception as exc:
        logger.exception("Chat failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/improve")
async def trigger_improvement(background: BackgroundTasks):
    """Manually trigger one improvement cycle."""
    background.add_task(_run_improvement_loop, [])
    return {"status": "improvement_loop_triggered"}


@app.get("/performance")
async def get_performance():
    """
    Return aggregate performance metrics sourced from Phoenix.
    Used by the dashboard accuracy chart.
    """
    analyzer: TraceFailureAnalyzer | None = _state.get("analyzer")
    loop: SelfImprovementLoop | None = _state.get("improvement_loop")

    by_version: dict = {}
    total = 0
    correct = 0

    if analyzer:
        try:
            # All traces (no eval filter) — pull recent 100 for metrics
            all_traces = await analyzer._t["get_traces"](
                project_name=config.PHOENIX_PROJECT_NAME,
                limit=100,
            )
            for t in all_traces.get("data", []):
                total += 1
                attrs = t.get("root_span", {}).get("attributes", {})
                pv    = attrs.get("matchmind.prompt_version", "v1")
                if pv not in by_version:
                    by_version[pv] = {"total": 0, "correct": 0}
                by_version[pv]["total"] += 1
                if attrs.get("matchmind.evaluated") and attrs.get("eval.accuracy") == "correct":
                    correct += 1
                    by_version[pv]["correct"] += 1
        except Exception as exc:
            logger.warning("Performance query failed: %s", exc)

    return {
        "total_predictions":  total,
        "accuracy_rate":      round(correct / total, 4) if total else 0.0,
        "by_prompt_version":  by_version,
        "improvement_cycles": loop._cycle_count if loop else 0,
        "last_improvement_at": None,
    }


@app.get("/health")
async def health():
    return {
        "status":  "healthy",
        "agent":   "matchmind",
        "version": "1.0.0",
        "model":   config.GEMINI_MODEL,
        "phoenix": config.PHOENIX_BASE_URL,
    }


# ── Background tasks ──────────────────────────────────────────────────────────

async def _run_improvement_loop(trigger_match_ids: list[str]) -> None:
    """
    Background task: run one self-improvement cycle.
    Errors are caught and logged — they must never crash the API server.
    """
    loop: SelfImprovementLoop | None = _state.get("improvement_loop")
    if not loop:
        logger.warning("Improvement loop not yet initialised — skipping cycle")
        return
    try:
        report = await loop.run(trigger_match_ids=trigger_match_ids)
        logger.info("Improvement cycle complete: %s", report.get("status"))
    except Exception as exc:
        logger.exception("Improvement loop error: %s", exc)
