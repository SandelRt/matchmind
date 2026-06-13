"""
Durable prediction store — the local source of truth.

Why this exists (June 2026 fix):
  The original design kept predictions, prompt versions, and cycle counts
  in scattered in-memory dicts. On Cloud Run those evaporate on scale-to-zero
  and diverge across instances. Worse, the improvement loop depended on
  querying Phoenix for failure traces via REST endpoints that don't exist,
  so it silently never ran.

  This store is now the primary data path: predictions are recorded at
  store time, evaluated when results arrive, and the improvement loop reads
  failures from here. Phoenix remains the observability/annotation layer
  (best-effort), not a hard dependency.

Persistence: JSON file with atomic writes (tmp + rename). On Cloud Run this
survives within an instance (pair with --min-instances=1 --max-instances=1).
For multi-instance production, swap _load/_save for Firestore — the
interface is deliberately small to make that a drop-in change.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("matchmind.store")

_EMPTY = {
    "predictions": {},        # match_id -> record
    "prompt_registry": {},    # version_tag -> prompt content
    "version_order": [],      # activation order, oldest first
    "active_version": None,   # None -> fall back to code baseline v1
    "rolled_back": [],        # versions deactivated by regression rollback
    "cycle_count": 0,
    "last_improvement_at": None,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionStore:
    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path or os.getenv("STORE_PATH", "/tmp/matchmind_store.json"))
        self._lock = threading.Lock()
        self._data = json.loads(json.dumps(_EMPTY))
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                loaded = json.loads(self._path.read_text())
                for key in _EMPTY:
                    if key in loaded:
                        self._data[key] = loaded[key]
                logger.info("Store loaded: %d predictions, active prompt=%s",
                            len(self._data["predictions"]), self._data["active_version"])
        except Exception as exc:
            logger.warning("Store load failed (%s) — starting fresh", exc)

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1))
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("Store save failed: %s", exc)

    # ── predictions ───────────────────────────────────────────────────────────

    def record_prediction(self, match_id: str, record: dict) -> None:
        with self._lock:
            record = dict(record)
            record.setdefault("stored_at", _utcnow())
            record.setdefault("actual_result", None)
            record.setdefault("evaluated", False)
            self._data["predictions"][match_id] = record
            self._save()

    def record_result(self, match_id: str, actual_score: str, direction: str) -> dict | None:
        with self._lock:
            rec = self._data["predictions"].get(match_id)
            if not rec:
                logger.warning("Result for unknown match_id=%s", match_id)
                return None
            rec["actual_result"] = actual_score
            rec["actual_direction"] = direction
            rec["result_at"] = _utcnow()
            self._save()
            return dict(rec)

    def attach_eval(self, match_id: str, eval_dict: dict) -> None:
        with self._lock:
            rec = self._data["predictions"].get(match_id)
            if not rec:
                return
            rec.update(eval_dict)
            rec["evaluated"] = True
            rec["evaluated_at"] = _utcnow()
            self._save()

    def get(self, match_id: str) -> dict | None:
        rec = self._data["predictions"].get(match_id)
        return dict(rec) if rec else None

    def get_failures(self, limit: int = 30) -> list[dict]:
        """Evaluated predictions whose direction was wrong, newest first."""
        rows = [
            dict(r) for r in self._data["predictions"].values()
            if r.get("evaluated") and r.get("accuracy") == "incorrect"
        ]
        rows.sort(key=lambda r: r.get("stored_at", ""), reverse=True)
        return rows[:limit]

    def totals(self) -> dict:
        preds = self._data["predictions"].values()
        evaluated = [r for r in preds if r.get("evaluated")]
        correct = [r for r in evaluated if r.get("accuracy") == "correct"]
        return {
            "total_predictions": len(self._data["predictions"]),
            "evaluated": len(evaluated),
            "correct": len(correct),
            "accuracy_rate": round(len(correct) / len(evaluated), 4) if evaluated else 0.0,
        }

    def accuracy_by_version(self) -> dict:
        out: dict[str, dict] = {}
        for r in self._data["predictions"].values():
            v = r.get("prompt_version", "v1")
            out.setdefault(v, {"total": 0, "evaluated": 0, "correct": 0})
            out[v]["total"] += 1
            if r.get("evaluated"):
                out[v]["evaluated"] += 1
                if r.get("accuracy") == "correct":
                    out[v]["correct"] += 1
        for v, s in out.items():
            s["accuracy_rate"] = round(s["correct"] / s["evaluated"], 3) if s["evaluated"] else None
        return out

    # ── prompt registry ───────────────────────────────────────────────────────

    def register_prompt(self, version_tag: str, content: str) -> None:
        with self._lock:
            self._data["prompt_registry"][version_tag] = content
            if version_tag not in self._data["version_order"]:
                self._data["version_order"].append(version_tag)
            self._data["active_version"] = version_tag
            self._save()
            logger.info("Prompt %s registered and activated", version_tag)

    def get_active_prompt(self) -> tuple[str, str] | None:
        v = self._data["active_version"]
        if v and v in self._data["prompt_registry"]:
            return v, self._data["prompt_registry"][v]
        return None

    def previous_version(self) -> str | None:
        order = [v for v in self._data["version_order"]
                 if v not in self._data["rolled_back"]]
        active = self._data["active_version"]
        if active in order:
            idx = order.index(active)
            if idx > 0:
                return order[idx - 1]
        # previous of the first registered version is the code baseline
        return "v1" if active else None

    def rollback(self) -> str | None:
        """Deactivate the active version; reactivate the one before it."""
        with self._lock:
            active = self._data["active_version"]
            if not active:
                return None
            prev = self.previous_version()
            self._data["rolled_back"].append(active)
            self._data["active_version"] = prev if prev != "v1" else None
            self._save()
            logger.warning("Rolled back prompt %s -> %s", active, prev or "v1 (baseline)")
            return prev or "v1"

    # ── improvement cycles ────────────────────────────────────────────────────

    def increment_cycle(self) -> int:
        with self._lock:
            self._data["cycle_count"] += 1
            self._data["last_improvement_at"] = _utcnow()
            self._save()
            return self._data["cycle_count"]

    @property
    def cycle_count(self) -> int:
        return self._data["cycle_count"]

    @property
    def last_improvement_at(self) -> str | None:
        return self._data["last_improvement_at"]


# Module-level singleton
store = PredictionStore()
