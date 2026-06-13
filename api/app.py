"""
MatchMind FastAPI server — deployed to Google Cloud Run.

Endpoints:
  POST /predict        Generate a match prediction            (auth required)
  POST /results        Submit actual match result             (auth required)
  POST /chat           Multi-turn conversation                (auth required)
  POST /improve        Manually trigger improvement cycle     (auth required)
  POST /demo           End-to-end demo                        (auth required)
  GET  /performance    Agent performance metrics              (public)
  GET  /health         Health check                           (public)
  GET  /               Dashboard                              (public)

June 2026 security fix: all mutating/LLM-invoking endpoints require an
X-API-Key header matching MATCHMIND_API_KEY. Without this, anyone with the
URL could drain the Gemini budget AND - far worse - POST fake results into
/results, which feeds the self-improvement loop that rewrites the agent's
own system prompt (unauthenticated prompt injection). If MATCHMIND_API_KEY
is unset, auth is disabled with a loud startup warning (local dev only).
"""
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import google.genai as genai
from google.genai import types as genai_types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent.agent import build_agent
from agent.config import config
from agent.prediction_store import store
from agent.prompts.templates import load_prompt_from_phoenix, get_active_version
from observability.tracing import setup_tracing, shutdown_tracing
from observability.evaluators import MatchPredictionEvaluator
from observability.phoenix_client import init_sync
from improvement.analyzer import TraceFailureAnalyzer
from improvement.loop import SelfImprovementLoop

logger = logging.getLogger("matchmind.api")
APP_NAME = "matchmind"
MAX_SESSIONS = 500
_state: dict = {}

# ── Auth ──────────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = config.MATCHMIND_API_KEY
    if not expected:
        return  # auth disabled (dev mode) — warned at startup
    if not key or key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MatchMind starting up...")
    if not config.MATCHMIND_API_KEY:
        logger.warning(
            "MATCHMIND_API_KEY is NOT set — all endpoints are UNAUTHENTICATED. "
            "Set it before exposing this service publicly."
        )

    setup_tracing(
        phoenix_api_key=config.PHOENIX_API_KEY,
        phoenix_base_url=config.PHOENIX_BASE_URL,
        project_name=config.PHOENIX_PROJECT_NAME,
    )
    logger.info("Phoenix tracing active -> %s", config.PHOENIX_BASE_URL)

    # Phoenix sync (annotations + prompt persistence) — best-effort layer
    sync = init_sync(
        api_key=config.PHOENIX_API_KEY,
        base_url=config.PHOENIX_BASE_URL,
        project_name=config.PHOENIX_PROJECT_NAME,
    )

    # Restore the latest improved prompt from Phoenix so self-improvements
    # survive cold starts and redeploys.
    try:
        restored = await load_prompt_from_phoenix(sync)
        if restored:
            logger.info("Prompt restored from Phoenix (version=%s)", get_active_version())
    except Exception as exc:
        logger.warning("Prompt restore skipped: %s", exc)

    agent = build_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    _state["agent"] = agent
    _state["runner"] = runner
    _state["session_service"] = session_service
    _state["sessions"] = OrderedDict()
    logger.info("ADK agent + Runner initialized (model: %s)", config.GEMINI_MODEL)

    gemini = genai.Client(api_key=config.GOOGLE_API_KEY)
    evaluator = MatchPredictionEvaluator()
    analyzer = TraceFailureAnalyzer(phoenix_sync=sync)
    improvement_loop = SelfImprovementLoop(
        gemini_client=gemini,
        analyzer=analyzer,
        evaluator=evaluator,
        project_name=config.PHOENIX_PROJECT_NAME,
        min_failures_to_trigger=config.IMPROVEMENT_TRIGGER_THRESHOLD,
        min_accuracy_delta=config.IMPROVEMENT_MIN_DELTA,
    )
    _state["evaluator"] = evaluator
    _state["gemini"] = gemini
    _state["analyzer"] = analyzer
    _state["improvement_loop"] = improvement_loop
    logger.info("MatchMind ready (threshold=%d)", config.IMPROVEMENT_TRIGGER_THRESHOLD)
    yield

    shutdown_tracing()
    logger.info("MatchMind shut down cleanly")


app = FastAPI(
    title="MatchMind",
    description="Self-improving WC2026 prediction agent. Powered by Gemini, traced by Arize Phoenix.",
    version="1.1.0",
    lifespan=lifespan,
)

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")


class PredictionRequest(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    match_date: str
    stage: str = "group"


class ResultRequest(BaseModel):
    match_id: str
    home_goals: int
    away_goals: int


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


async def _run_agent(prompt: str, session_id: str = "default") -> str:
    runner = _state["runner"]
    session_service = _state["session_service"]
    sessions: OrderedDict = _state["sessions"]

    if session_id in sessions:
        session = sessions[session_id]
        sessions.move_to_end(session_id)
    else:
        session = await session_service.create_session(app_name=APP_NAME, user_id="default")
        sessions[session_id] = session
        while len(sessions) > MAX_SESSIONS:  # LRU eviction — was unbounded
            sessions.popitem(last=False)

    user_message = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
    final_text = ""
    async for event in runner.run_async(
        user_id="default", session_id=session.id, new_message=user_message
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break
    return final_text


@app.get("/")
async def dashboard():
    index = _frontend / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "MatchMind API", "docs": "/docs"})


@app.post("/predict", dependencies=[Security(require_api_key)])
async def predict_match(req: PredictionRequest):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    prompt = (
        f"Predict the World Cup match: {req.home_team} vs {req.away_team} "
        f"on {req.match_date} (stage: {req.stage}, match_id: {req.match_id}). "
        f"Follow the mandatory 6-step prediction protocol. "
        f"Store the prediction with match_id '{req.match_id}' "
        f"and prompt_version '{get_active_version()}'."
    )
    try:
        result = await _run_agent(prompt, session_id=f"predict_{req.match_id}")
        return {"status": "ok", "match_id": req.match_id, "result": result}
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/results", dependencies=[Security(require_api_key)])
async def submit_result(req: ResultRequest, background: BackgroundTasks):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    from agent.tools.match_data import record_result as _rec
    from agent.tools.prediction import process_match_result

    _rec(req.match_id, req.home_goals, req.away_goals)
    actual_score = f"{req.home_goals}-{req.away_goals}"

    # Let the agent narrate/update via its tool...
    update_prompt = (
        f"The match with ID '{req.match_id}' has ended. "
        f"Final score: {actual_score}. "
        f"Call update_prediction_with_result() to record this result."
    )
    try:
        await _run_agent(update_prompt, session_id=f"result_{req.match_id}")
    except Exception as exc:
        logger.warning("Agent result-update run failed (%s) — using direct path", exc)

    # ...but never DEPEND on the LLM calling the tool: if the record is
    # still unevaluated, run the same pipeline directly.
    rec = store.get(req.match_id)
    eval_summary = None
    if rec and not rec.get("evaluated"):
        result = await process_match_result(req.match_id, req.home_goals, req.away_goals)
        eval_summary = result.get("accuracy")
    elif rec:
        eval_summary = rec.get("accuracy")

    background.add_task(_run_improvement_loop, [req.match_id])
    return {
        "status": "result_recorded",
        "match_id": req.match_id,
        "actual_score": actual_score,
        "accuracy": eval_summary,
        "improvement_loop_queued": True,
    }


@app.post("/chat", dependencies=[Security(require_api_key)])
async def chat(req: ChatRequest):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    try:
        result = await _run_agent(req.message, session_id=req.session_id)
        return {"response": result, "session_id": req.session_id}
    except Exception as exc:
        logger.exception("Chat failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/improve", dependencies=[Security(require_api_key)])
async def trigger_improvement(background: BackgroundTasks):
    background.add_task(_run_improvement_loop, [])
    return {"status": "improvement_loop_triggered"}


@app.post("/demo", dependencies=[Security(require_api_key)])
async def run_demo():
    """End-to-end demo: predict 3 matches, submit results, run improvement cycle."""
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    from agent.tools.match_data import record_result as _rec
    from agent.tools.prediction import process_match_result

    demo_matches = [
        {"match_id": "WC26_G1", "home_team": "Argentina", "away_team": "Chile",
         "match_date": "2026-06-14T20:00:00Z", "stage": "group"},
        {"match_id": "WC26_B1", "home_team": "Spain", "away_team": "Croatia",
         "match_date": "2026-06-12T18:00:00Z", "stage": "group"},
        {"match_id": "WC26_C1", "home_team": "Germany", "away_team": "Japan",
         "match_date": "2026-06-12T20:00:00Z", "stage": "group"},
    ]
    demo_results = [
        {"match_id": "WC26_G1", "home_goals": 3, "away_goals": 0},
        {"match_id": "WC26_B1", "home_goals": 2, "away_goals": 1},
        {"match_id": "WC26_C1", "home_goals": 2, "away_goals": 2},
    ]

    steps = []

    for m in demo_matches:
        prompt = (
            f"Predict: {m['home_team']} vs {m['away_team']} "
            f"(stage={m['stage']}, match_id={m['match_id']}). "
            f"Follow the full 6-step prediction protocol. "
            f"Store with store_prediction() using prompt_version '{get_active_version()}'."
        )
        try:
            prediction = await _run_agent(prompt, session_id=f"demo_{m['match_id']}")
            steps.append({
                "step": "predict",
                "match_id": m["match_id"],
                "match": f"{m['home_team']} vs {m['away_team']}",
                "status": "ok",
                "prediction_summary": (prediction or "")[:300],
            })
        except Exception as exc:
            steps.append({"step": "predict", "match_id": m["match_id"],
                          "status": "error", "error": str(exc)})

    for r in demo_results:
        _rec(r["match_id"], r["home_goals"], r["away_goals"])
        try:
            result = await process_match_result(
                r["match_id"], r["home_goals"], r["away_goals"]
            )
            steps.append({
                "step": "result",
                "match_id": r["match_id"],
                "actual": f"{r['home_goals']}-{r['away_goals']}",
                "accuracy": result.get("accuracy"),
                "status": result.get("status", "recorded"),
            })
        except Exception as exc:
            steps.append({"step": "result", "match_id": r["match_id"],
                          "status": "error", "error": str(exc)})

    loop = _state.get("improvement_loop")
    if loop:
        try:
            ids = [r["match_id"] for r in demo_results]
            report = await loop.run(trigger_match_ids=ids)
            steps.append({"step": "improve", "status": report.get("status", "unknown"),
                          "report": report})
        except Exception as exc:
            steps.append({"step": "improve", "status": "error", "error": str(exc)})
    else:
        steps.append({"step": "improve", "status": "skipped"})

    return {
        "demo": "complete",
        "matches_predicted": len(demo_matches),
        "results_submitted": len(demo_results),
        "steps": steps,
    }


@app.get("/performance")
async def get_performance():
    """
    Performance metrics — now read from the durable PredictionStore (the
    same records the evaluators write) instead of misreading root-span
    attributes from a Phoenix endpoint that never returned data.
    """
    totals = store.totals()
    return {
        "total_predictions": totals["total_predictions"],
        "evaluated": totals["evaluated"],
        "accuracy_rate": totals["accuracy_rate"],
        "by_prompt_version": store.accuracy_by_version(),
        "active_prompt_version": get_active_version(),
        "improvement_cycles": store.cycle_count,
        "last_improvement_at": store.last_improvement_at,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "matchmind",
        "version": "1.1.0",
        "model": config.GEMINI_MODEL,
        "phoenix": config.PHOENIX_BASE_URL,
        "auth": "enabled" if config.MATCHMIND_API_KEY else "DISABLED (dev mode)",
        "active_prompt_version": get_active_version(),
    }


async def _run_improvement_loop(trigger_match_ids: list) -> None:
    loop = _state.get("improvement_loop")
    if not loop:
        logger.warning("Improvement loop not initialised — skipping")
        return
    try:
        report = await loop.run(trigger_match_ids=trigger_match_ids)
        logger.info("Improvement cycle complete: %s", report.get("status"))
    except Exception as exc:
        logger.exception("Improvement loop error: %s", exc)
