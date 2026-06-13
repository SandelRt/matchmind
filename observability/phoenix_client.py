"""
Phoenix sync layer — wraps the official `arize-phoenix-client` SDK.

June 2026 fix: the previous version hand-rolled REST calls against
endpoints that don't exist in the Phoenix API (e.g. GET /v1/traces with an
eval filter_condition) and swallowed every error into {"data": []}. The
improvement loop therefore always saw zero failures and silently no-oped.

This rewrite:
  - uses the official AsyncClient (correct endpoints, correct auth)
  - uploads eval results as span annotations  (client.spans.log_span_annotations)
  - persists prompt versions to Phoenix Prompt Management (client.prompts.create)
  - fetches the latest prompt version at startup     (client.prompts.get)
  - is explicitly BEST-EFFORT: the local PredictionStore is the source of
    truth; every Phoenix failure is logged loudly and returns None/False
    instead of pretending to be data.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger("matchmind.observability.phoenix")

PROMPT_NAME = "match_prediction_prompt"


class PhoenixSync:
    def __init__(self, api_key: str, base_url: str, project_name: str) -> None:
        self.project = project_name
        self.enabled = False
        self._client = None
        if not api_key:
            logger.info("PHOENIX_API_KEY not set — Phoenix sync disabled")
            return
        try:
            from phoenix.client import AsyncClient
            self._client = AsyncClient(base_url=base_url, api_key=api_key)
            self.enabled = True
            logger.info("Phoenix sync enabled -> %s (project=%s)", base_url, project_name)
        except Exception as exc:
            logger.warning("Phoenix client unavailable (%s) — sync disabled", exc)

    # ── eval annotations ──────────────────────────────────────────────────────

    async def log_eval_annotations(self, span_id: str, eval_result: dict) -> bool:
        """
        Attach eval results to the original prediction span as annotations.
        This replaces the broken pattern of trying to mutate an already-exported
        span (OTel spans are immutable) — annotations are Phoenix's mechanism
        for exactly this.

        eval_result keys used: accuracy, accuracy_score, calibration,
        calibration_score, reasoning_quality, reasoning_score, composite_score.
        """
        if not self.enabled or not span_id:
            return False
        annotations = []
        for name, label_key, score_key in (
            ("accuracy", "accuracy", "accuracy_score"),
            ("calibration", "calibration", "calibration_score"),
            ("reasoning_quality", "reasoning_quality", "reasoning_score"),
        ):
            if label_key in eval_result:
                annotations.append({
                    "name": name,
                    "annotator_kind": "CODE",
                    "span_id": span_id,
                    "result": {
                        "label": str(eval_result[label_key]),
                        "score": float(eval_result.get(score_key, 0.0)),
                    },
                })
        if "composite_score" in eval_result:
            annotations.append({
                "name": "composite",
                "annotator_kind": "CODE",
                "span_id": span_id,
                "result": {"score": float(eval_result["composite_score"])},
            })
        try:
            await self._client.spans.log_span_annotations(span_annotations=annotations)
            logger.info("Phoenix: %d annotations logged for span %s", len(annotations), span_id)
            return True
        except Exception as exc:
            logger.warning("Phoenix annotation upload failed for span %s: %s", span_id, exc)
            return False

    # ── prompt management ─────────────────────────────────────────────────────

    async def push_prompt_version(
        self,
        content: str,
        description: str,
        model_name: str,
        name: str = PROMPT_NAME,
    ) -> Optional[str]:
        """Persist a prompt version to Phoenix Prompt Management. Returns version id."""
        if not self.enabled:
            return None
        try:
            from phoenix.client.types import PromptVersion
            version = PromptVersion(
                [{"role": "system", "content": content}],
                model_name=model_name,
                model_provider="GOOGLE",
                template_format="NONE",
                description=description,
            )
            created = await self._client.prompts.create(
                name=name, version=version, prompt_description=description,
            )
            vid = getattr(created, "id", None)
            logger.info("Phoenix: prompt version pushed (id=%s)", vid)
            return vid
        except Exception as exc:
            logger.warning("Phoenix prompt push failed: %s", exc)
            return None

    async def fetch_latest_prompt(self, name: str = PROMPT_NAME) -> Optional[str]:
        """Fetch latest prompt version content. Used at startup to survive restarts."""
        if not self.enabled:
            return None
        try:
            version = await self._client.prompts.get(prompt_identifier=name)
            return _extract_prompt_text(version)
        except Exception as exc:
            logger.info("Phoenix: no stored prompt fetched (%s) — using local", exc)
            return None


def _extract_prompt_text(version: Any) -> Optional[str]:
    """Pull the system-message text out of a PromptVersion, defensively."""
    try:
        template = getattr(version, "_template", None) or {}
        messages = template.get("messages", []) if isinstance(template, dict) else []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]
    except Exception as exc:
        logger.warning("Could not extract prompt text: %s", exc)
    return None


# ── Module singleton ──────────────────────────────────────────────────────────

_sync: Optional[PhoenixSync] = None


def init_sync(api_key: str, base_url: str, project_name: str) -> PhoenixSync:
    global _sync
    _sync = PhoenixSync(api_key=api_key, base_url=base_url, project_name=project_name)
    return _sync


def get_sync() -> Optional[PhoenixSync]:
    return _sync
