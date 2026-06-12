"""
MatchMind FastAPI server — deployed to Google Cloud Run.

Endpoints:
  POST /predict        Generate a match prediction
  POST /results        Submit actual match result (triggers improvement loop)
  POST /chat           Multi-turn conversation
  POST /improve        Manually trigger improvement cycle
  POST /demo           End-to-end demo: predict 3 matches + improve
  GET  /performance    Agent performance metrics
  GET  /health         Health check
  GET  /              Dashboard (serves frontend/index.html)
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import google.genai as genai
from google.genai import types as genai_types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent.agent import build_agent
from agent.config import config
from observability.tracing import setup_tracing, shutdown_tracing
from observability.evaluators import MatchPredictionEvaluator
from observability.phoenix_client import build_phoenix_tools
from improvement.analyzer import TraceFailureAnalyzer
from improvement.loop import SelfImprovementLoop

logger = logging.getLogger("matchmind.api")
APP_NAME = "matchmind"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MatchMind starting up...")
    setup_tracing(
        phoenix_api_key=config.PHOENIX_API_KEY,
        phoenix_base_url=config.PHOENIX_BASE_URL,
        project_name=config.PHOENIX_PROJECT_NAME,
    )
    logger.info("Phoenix tracing active -> %s", config.PHOENIX_BASE_URL)

    agent = build_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    _state["agent"] = agent
    _state["runner"] = runner
    _state["session_service"] = session_service
    _state["sessions"] = {}
    logger.info("ADK agent + Runner initialized (model: %s)", config.GEMINI_MODEL)

    gemini = genai.Client()
    evaluator = MatchPredictionEvaluator()
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
    version="1.0.0",
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
    sessions = _state["sessions"]

    if session_id in sessions:
        session = sessions[session_id]
    else:
        session = await session_service.create_session(app_name=APP_NAME, user_id="default")
        sessions[session_id] = session

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


@app.post("/predict")
async def predict_match(req: PredictionRequest):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    prompt = (
        f"Predict the World Cup match: {req.home_team} vs {req.away_team} "
        f"on {req.match_date} (stage: {req.stage}, match_id: {req.match_id}). "
        f"Follow the mandatory 6-step prediction protocol. "
        f"Store the prediction with match_id '{req.match_id}'."
    )
    try:
        result = await _run_agent(prompt, session_id=f"predict_{req.match_id}")
        return {"status": "ok", "match_id": req.match_id, "result": result}
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/results")
async def submit_result(req: ResultRequest, background: BackgroundTasks):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    from agent.tools.match_data import record_result as _rec
    _rec(req.match_id, req.home_goals, req.away_goals)
    actual_score = f"{req.home_goals}-{req.away_goals}"
    update_prompt = (
        f"The match with ID '{req.match_id}' has ended. "
        f"Final score: {actual_score}. "
        f"Call update_prediction_with_result() to record this result."
    )
    await _run_agent(update_prompt, session_id=f"result_{req.match_id}")
    background.add_task(_run_improvement_loop, [req.match_id])
    return {
        "status": "result_recorded",
        "match_id": req.match_id,
        "actual_score": actual_score,
        "improvement_loop_queued": True,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    try:
        result = await _run_agent(req.message, session_id=req.session_id)
        return {"response": result, "session_id": req.session_id}
    except Exception as exc:
        logger.exception("Chat failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/improve")
async def trigger_improvement(background: BackgroundTasks):
    background.add_task(_run_improvement_loop, [])
    return {"status": "improvement_loop_triggered"}


@app.post("/demo")
async def run_demo():
    """End-to-end demo: predict 3 matches, submit results, run improvement cycle."""
    if not _state.get("runner"):
        raise HTTPException(503, "Agent not ready")
    from agent.tools.match_data import record_result as _rec

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
            f"Store with store_prediction()."
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
        update_prompt = (
            f"Match {r['match_id']} ended {r['home_goals']}-{r['away_goals']}. "
            f"Call update_prediction_with_result()."
        )
        try:
            await _run_agent(update_prompt, session_id=f"demo_result_{r['match_id']}")
            steps.append({
                "step": "result",
                "match_id": r["match_id"],
                "actual": f"{r['home_goals']}-{r['away_goals']}",
                "status": "recorded",
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
    analyzer = _state.get("analyzer")
    loop = _state.get("improvement_loop")
    by_version: dict = {}
    total = 0
    correct = 0
    if analyzer:
        try:
            all_traces = await analyzer._t["get_traces"](
                project_name=config.PHOENIX_PROJECT_NAME, limit=100
            )
            for t in all_traces.get("data", []):
                total += 1
                attrs = t.get("root_span", {}).get("attributes", {})
                pv = attrs.get("matchmind.prompt_version", "v1")
                if pv not in by_version:
                    by_version[pv] = {"total": 0, "correct": 0}
                by_version[pv]["total"] += 1
                if attrs.get("matchmind.evaluated") and attrs.get("eval.accuracy") == "correct":
                    correct += 1
                    by_version[pv]["correct"] += 1
        except Exception as exc:
            logger.warning("Performance query failed: %s", exc)
    return {
        "total_predictions": total,
        "accuracy_rate": round(correct / total, 4) if total else 0.0,
        "by_prompt_version": by_version,
        "improvement_cycles": getattr(loop, "_cycle_count", 0),
        "last_improvement_at": None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "matchmind",
        "version": "1.0.0",
        "model": config.GEMINI_MODEL,
        "phoenix": config.PHOENIX_BASE_URL,
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
