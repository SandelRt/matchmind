"""
Central configuration — all env vars in one place.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Google Cloud / Gemini ─────────────────────────────────────────────────
    GOOGLE_CLOUD_PROJECT: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    GOOGLE_CLOUD_LOCATION: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # ── Arize Phoenix ─────────────────────────────────────────────────────────
    PHOENIX_API_KEY: str = field(
        default_factory=lambda: os.getenv("PHOENIX_API_KEY", "")
    )
    PHOENIX_BASE_URL: str = field(
        default_factory=lambda: os.getenv("PHOENIX_BASE_URL", "https://app.phoenix.arize.com")
    )
    PHOENIX_PROJECT_NAME: str = field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "matchmind")
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_HOST: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    APP_PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ── Improvement Loop ──────────────────────────────────────────────────────
    # Min failures before triggering improvement cycle
    IMPROVEMENT_TRIGGER_THRESHOLD: int = field(
        default_factory=lambda: int(os.getenv("IMPROVEMENT_TRIGGER_THRESHOLD", "1"))
    )
    # Min accuracy delta to deploy a new prompt version
    IMPROVEMENT_MIN_DELTA: float = field(
        default_factory=lambda: float(os.getenv("IMPROVEMENT_MIN_DELTA", "0.05"))
    )


config = Config()
