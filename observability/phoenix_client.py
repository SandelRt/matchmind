"""
Phoenix REST client wrappers.

Provides the same async-callable interface that TraceFailureAnalyzer and
SelfImprovementLoop expect, but backed by the Phoenix HTTP API instead of
the MCP stdio subprocess.

This lets the improvement loop run independently of the agent's MCP session
— it calls Phoenix directly over HTTPS while the ADK agent uses the MCP
server for in-conversation self-introspection.

Usage:
    tools = build_phoenix_tools(api_key=..., base_url=..., project=...)
    analyzer = TraceFailureAnalyzer(phoenix_tools=tools)
"""
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("matchmind.observability.phoenix_client")

# ── Low-level HTTP client ──────────────────────────────────────────────────────

class PhoenixHTTPClient:
    """Thin async wrapper around the Arize Phoenix REST API."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {
            "api_key":      api_key,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._base}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._headers, params=params or {})
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        url = f"{self._base}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url, headers=self._headers, content=json.dumps(body)
            )
            resp.raise_for_status()
            return resp.json()

    # ── Phoenix API methods ───────────────────────────────────────────────────

    async def get_traces(
        self,
        project_name: str,
        filter_condition: str | None = None,
        limit: int = 20,
        sort_by: str = "start_time",
        sort_direction: str = "desc",
    ) -> dict:
        """Query traces for a project, optionally filtered by eval labels."""
        params: dict[str, Any] = {
            "project_name": project_name,
            "limit":        limit,
            "sort_by":      sort_by,
            "sort_direction": sort_direction,
        }
        if filter_condition:
            params["filter_condition"] = filter_condition
        try:
            return await self._get("/v1/traces", params)
        except Exception as exc:
            logger.warning("get_traces failed: %s — returning empty", exc)
            return {"data": []}

    async def get_spans(
        self,
        project_name: str,
        trace_id: str | None = None,
        limit: int = 50,
    ) -> dict:
        params: dict[str, Any] = {"project_name": project_name, "limit": limit}
        if trace_id:
            params["trace_id"] = trace_id
        try:
            return await self._get("/v1/spans", params)
        except Exception as exc:
            logger.warning("get_spans failed: %s — returning empty", exc)
            return {"data": []}

    async def get_prompts(self, project_name: str) -> dict:
        try:
            return await self._get(
                "/v1/prompts", {"project_name": project_name}
            )
        except Exception as exc:
            logger.warning("get_prompts failed: %s — returning empty", exc)
            return {"data": []}

    async def create_prompt(
        self,
        project_name: str,
        name: str,
        version: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        body = {
            "project_name": project_name,
            "name":         name,
            "version":      version,
            "content":      content,
            "description":  description,
            "tags":         tags or [],
        }
        try:
            return await self._post("/v1/prompts", body)
        except Exception as exc:
            logger.warning("create_prompt failed: %s — returning stub", exc)
            return {"id": None, "version": version}

    async def get_experiments(
        self,
        project_name: str,
        dataset_name: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"project_name": project_name}
        if dataset_name:
            params["dataset_name"] = dataset_name
        try:
            return await self._get("/v1/experiments", params)
        except Exception as exc:
            logger.warning("get_experiments failed: %s — returning empty", exc)
            return {"data": []}

    async def list_projects(self) -> dict:
        try:
            return await self._get("/v1/projects")
        except Exception as exc:
            logger.warning("list_projects failed: %s", exc)
            return {"data": []}


# ── Tool dict builder ─────────────────────────────────────────────────────────

def build_phoenix_tools(
    api_key: str,
    base_url: str,
    project_name: str,  # noqa: F841 – kept for future per-project filtering
) -> dict:
    """
    Return the tool-callable dict expected by TraceFailureAnalyzer.

    Each value is an async callable with the same signature as the
    corresponding Phoenix MCP tool.
    """
    client = PhoenixHTTPClient(api_key=api_key, base_url=base_url)

    return {
        "get_traces":       client.get_traces,
        "get_spans":        client.get_spans,
        "get_prompts":      client.get_prompts,
        "create_prompt":    client.create_prompt,
        "get_experiments":  client.get_experiments,
        "list_projects":    client.list_projects,
    }
