# MatchMind v1.1.0 — June 2026 Fix Release

Code review found that the headline feature — the self-improvement loop —
was wired so it could never actually run, and the service was deployed with
no authentication. This release fixes the loop end-to-end, adds auth, and
adds a regression guard so a bad self-rewrite can't become permanent.

## Critical fixes

### 1. Live prompt updates now actually happen
`LlmAgent` was built with `instruction=get_active_prediction_prompt()` — a
string frozen at startup. New prompt versions registered by the loop never
reached the agent. The agent now uses an ADK **InstructionProvider**
(callable, re-evaluated every run), verified against ADK 1.3.0.
→ `agent/agent.py`, `agent/prompts/templates.py`

### 2. The improvement loop has real data now
The old Phoenix REST wrappers called endpoints that don't exist
(`GET /v1/traces` + eval filters) and swallowed every error into
`{"data": []}` — so the loop always saw 0 failures and silently skipped.
Predictions/results/evals now live in a durable **PredictionStore**
(`agent/prediction_store.py`, JSON + atomic writes), which is the loop's
source of truth. → `improvement/analyzer.py`, `improvement/loop.py`

### 3. Evals are computed on result arrival and written back to Phoenix
Evaluation now runs immediately when a result arrives (deterministic,
no LLM cost) and uploads **span annotations** to the original prediction
span via the official `arize-phoenix-client` SDK — replacing the broken
attempt to mutate already-exported (immutable) OTel spans.
→ `agent/tools/prediction.py`, `observability/phoenix_client.py`
The API no longer depends on the LLM remembering to call the update tool:
`/results` falls back to `process_match_result()` directly.

### 4. Authentication
All mutating/LLM endpoints (`/predict`, `/results`, `/chat`, `/improve`,
`/demo`) require `X-API-Key` matching the `MATCHMIND_API_KEY` env var.
Previously anyone with the URL could drain the Gemini budget — or POST fake
results into the loop that rewrites the agent's own system prompt.
If unset, auth is disabled with a loud startup warning (dev only).

### 5. Regression guard + rollback
`min_accuracy_delta` was accepted and never used; prompts activated
unconditionally. The loop now runs a regression check each cycle: if the
active version is measurably worse than its nearest evidenced predecessor,
it rolls back before generating anything new.

## Other fixes
- Prompt versions are pushed to Phoenix Prompt Management and **restored at
  startup**, so improvements survive cold starts/redeploys.
- `/performance` reads the store (was: misread root-span attrs from a dead
  endpoint; could never count a correct prediction).
- Generated-prompt sanity check (rejects empty/short rewrites).
- Hardcoded `gemini-2.0-flash` in the meta-prompt call → `config.GEMINI_MODEL`.
- Session dict is LRU-capped at 500 (was unbounded).
- `datetime.utcnow()` (deprecated on 3.12) → `datetime.now(timezone.utc)`.
- Removed unused `wc26-mcp` prewarm from Dockerfile (never wired in code).
- Cloud Run pinned to one warm instance (per-instance state); swap
  PredictionStore to Firestore before scaling out.
- `arize-phoenix-client` pinned `>=2.9.0` (the API surface used here).

## Deploying
```bash
# one-time: create the API key secret
openssl rand -hex 24 | gcloud secrets create matchmind-api-key --data-file=-

# attach it to the service (alongside existing google-api-key / phoenix-api-key)
gcloud run services update matchmind --region us-central1 \
  --set-secrets=MATCHMIND_API_KEY=matchmind-api-key:latest

# build + deploy
gcloud builds submit --config cloudbuild.yaml
```
Callers must now send `X-API-Key: <key>` on POST endpoints.

## Tests
`python tests/run_tests.py` — 28 checks covering the store, eval pipeline,
failure analysis, live prompt activation, the full improvement cycle (fake
Gemini), regression rollback, and restart durability. No network/ADK needed.

## Known limitations (deliberate scope)
- JSON store is per-instance: fine at max-instances=1; use Firestore to scale.
- Regression guard needs ≥3 evaluated predictions on the active version.
- The dashboard frontend still polls the old metric names where applicable.
- Model remains `gemini-2.0-flash` for parity with the deployed service —
  upgrading to a current Flash model is recommended (one env var).
