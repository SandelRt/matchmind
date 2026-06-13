"""
Local development runner for MatchMind.

Starts the FastAPI server with hot-reload disabled for stability.
Use this for local testing before deploying to Cloud Run.

Usage:
    cd matchmind
    python -m agent.main
    # or
    python agent/main.py
"""
import uvicorn
from agent.config import config

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=False,
    )
