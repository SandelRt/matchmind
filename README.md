# MatchMind — Self-Improving World Cup 2026 Prediction Agent

**Apache 2.0 License** · Built for the Google Cloud Rapid Agent Hackathon · Arize Partner Track

> MatchMind watches itself fail, then rewrites its own instructions.
> Every wrong prediction makes the next one better.

---

## What It Does

MatchMind is a World Cup 2026 match prediction agent that uses **Arize Phoenix** to observe every decision it makes — and then uses those observations to autonomously improve itself.

The self-improvement loop runs automatically after each match result arrives:

```
Match Result Submitted
        ↓
Phoenix traces queried (failures only)
        ↓
LLM evaluators score accuracy + calibration + reasoning quality
        ↓
Failure patterns extracted (overconfidence, missing injury checks, etc.)
        ↓
Gemini rewrites the system prompt to fix those patterns
        ↓
New prompt version persisted to Phoenix Prompt Management
        ↓
Agent picks up new prompt immediately (no redeploy required)
```

No human in the loop. The agent gets measurably better across every tournament round.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Cloud Run                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  FastAPI     │    │   Google ADK LlmAgent (Gemini)       │   │
│  │  /predict    │───▶│                                      │   │
│  │  /results    │    │  ┌──────────────┐ ┌───────────────┐  │   │
│  │  /chat       │    │  │ Phoenix MCP  │ │  WC26 MCP     │  │   │
│  │  /improve    │    │  │ (self-trace) │ │ (live data)   │  │   │
│  │  /performance│    │  └──────────────┘ └───────────────┘  │   │
│  └──────────────┘    │  ┌──────────────────────────────────┐│   │
│         │            │  │  Custom FunctionTools            ││   │
│         │            │  │  store_prediction • get_form     ││   │
│         │            │  │  get_injuries • head_to_head     ││   │
│         │            │  └──────────────────────────────────┘│   │
│         │            └──────────────────────────────────────┘   │
│         │                        │                               │
│         │           OpenInference auto-instrumentation           │
│         │                        ↓                               │
│  ┌──────▼──────────────────────────────────────────────────┐    │
│  │          Self-Improvement Loop                           │    │
│  │   TraceFailureAnalyzer → SelfImprovementLoop             │    │
│  │   (pulls failures via Phoenix REST API)                  │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Arize Phoenix      │
                   │   (Cloud SaaS)       │
                   │                      │
                   │  • Traces & Spans    │
                   │  • LLM Evaluations   │
                   │  • Prompt Versions   │
                   │  • Experiments       │
                   └─────────────────────┘
```

### Key components

| Component | Purpose |
|-----------|---------|
| `agent/agent.py` | ADK `LlmAgent` with Phoenix MCP + WC26 MCP + 9 custom tools |
| `observability/tracing.py` | OpenInference auto-instrumentation → Phoenix Cloud |
| `observability/evaluators.py` | Accuracy + calibration + reasoning quality scorers |
| `improvement/analyzer.py` | Queries Phoenix for failure traces + extracts patterns |
| `improvement/loop.py` | 6-step self-improvement cycle (Gemini rewrites the prompt) |
| `agent/prompts/templates.py` | Versioned system prompts; fetches active from Phoenix |
| `api/app.py` | FastAPI server; triggers loop after each result |
| `frontend/index.html` | Dashboard: accuracy chart, version table, chat UI |

---

## Quickstart (Local)

### Prerequisites

- Python 3.12+
- Node.js 20+ (for MCP servers via npx)
- Google Cloud project with Vertex AI enabled
- Arize Phoenix Cloud account (free): https://app.phoenix.arize.com

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/matchmind.git
cd matchmind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   GOOGLE_CLOUD_PROJECT=your-project-id
#   GOOGLE_CLOUD_LOCATION=us-central1
#   PHOENIX_API_KEY=your-phoenix-api-key
```

Get your Phoenix API key at: https://app.phoenix.arize.com → Settings → API Keys

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 4. Start the server

```bash
python agent/main.py
# Server starts at http://localhost:8080
```

### 5. Open the dashboard

```
http://localhost:8080
```

### 6. Run the demo

```bash
python scripts/demo.py --api http://localhost:8080
```

This runs 14 simulated match predictions across 3 rounds and demonstrates
the accuracy improvement curve as the self-improvement loop fires.

---

## API Reference

```
POST /predict     Generate a match prediction
POST /results     Submit actual result (triggers improvement loop)
POST /chat        Chat with the agent
POST /improve     Manually trigger one improvement cycle
GET  /performance Accuracy metrics by prompt version
GET  /health      Health check
GET  /            Dashboard UI
```

### Example: Predict a match

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "match_id":   "WC26_QF_01",
    "home_team":  "Brazil",
    "away_team":  "France",
    "match_date": "2026-07-03T20:00:00Z",
    "stage":      "quarter_final"
  }'
```

### Example: Submit the result

```bash
curl -X POST http://localhost:8080/results \
  -H "Content-Type: application/json" \
  -d '{"match_id": "WC26_QF_01", "home_goals": 2, "away_goals": 1}'
```

---

## Deploy to Google Cloud Run

```bash
# One-time: create the Phoenix API key secret
gcloud secrets create phoenix-api-key --data-file=- <<< "YOUR_PHOENIX_API_KEY"

# Build and deploy
gcloud builds submit --config cloudbuild.yaml
```

The `cloudbuild.yaml` builds the container, pushes to GCR, and deploys
to Cloud Run with the Phoenix API key injected from Secret Manager.

---

## How the Self-Improvement Loop Works

After every match result:

1. **Fetch** — queries Phoenix for traces labelled `accuracy = incorrect`
2. **Evaluate** — scores each failure on accuracy, calibration, reasoning quality
3. **Analyse** — computes pattern rates: overconfidence, missing injury checks, shallow reasoning
4. **Generate** — sends current prompt + failure analysis to Gemini with conditional rules:
   - If injury miss rate > 30% → add hard STOP requiring injury check
   - If overconfidence rate > 25% → strengthen calibration rules
   - If shallow reasoning > 30% → enforce 80-word minimum reasoning
5. **Persist** — saves new prompt version to Phoenix Prompt Management with version tag `vYYYYMMDD_HHMM`
6. **Activate** — new prompt is live immediately; no redeploy needed

This creates a measurable improvement curve across tournament rounds, visible in the dashboard.

---

## What Makes This Different

Most prediction agents are **static**. MatchMind is **self-modifying**.

The agent doesn't just use Arize Phoenix for passive monitoring — it actively reads its own traces via the **Phoenix MCP server**, reasons over its failure history during predictions ("why did I get this wrong last time?"), and then rewrites its own instructions via Gemini when patterns emerge.

Observable at every level:
- Every LLM call, tool call, and agent step is a Phoenix span
- Every prediction is an annotated trace with 15+ custom attributes
- Every improvement cycle is a traced span with before/after metrics
- Every prompt version is tracked in Phoenix Prompt Management

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 2.0 Flash (Vertex AI) |
| Agent runtime | Google ADK 1.3.0 |
| Observability | Arize Phoenix Cloud + OpenInference |
| Live data | WC26 MCP (18 World Cup tools) |
| Self-introspection | Phoenix MCP Server |
| API | FastAPI + uvicorn |
| Deployment | Google Cloud Run |
| CI/CD | Cloud Build |

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE)
