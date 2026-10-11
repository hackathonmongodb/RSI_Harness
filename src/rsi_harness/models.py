"""Annotated collection shapes for NL-to-MongoDB prompting (MongoDB docs recommend native types)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolAccess(BaseModel):
    """Which tools the agent may call for this harness generation."""

    allowed: list[str] = Field(default_factory=list)
    denied: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)


class HarnessVersion(BaseModel):
    """A complete agent environment: rules, context policy, guardrails, tool access."""

    version: int
    status: Literal["candidate", "production", "retired"] = "candidate"
    parent_version: int | None = None
    rules: list[str] = Field(default_factory=list)
    context_policy: str = ""
    guardrails: list[str] = Field(default_factory=list)
    tool_access: ToolAccess = Field(default_factory=ToolAccess)
    mutation_reason: str = ""
    avg_score: float | None = None
    created_at: datetime


class EvalRun(BaseModel):
    """Hard metric signal after a task. Drives recursive self-improvement."""

    harness_version: int
    task: str
    score: float
    latency_ms: int
    tokens_used: int
    guardrail_violations: int
    notes: str = ""
    created_at: datetime


class Memory(BaseModel):
    """Durable memory across long-horizon sessions (not a vector RAG app)."""

    tier: Literal["user", "project", "team", "agent"]
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class QueryTrace(BaseModel):
    """Every natural-language datastore access is recorded for the demo and judges."""

    natural_language: str
    collection: str
    mongosh: str
    operation: Literal["find", "aggregate"]
    executable: dict[str, Any]
    result_count: int
    latency_ms: int
    created_at: datetime


COLLECTION_DOCS = {
    "harness_versions": "Versioned agent harnesses. Later versions evolve from earlier ones.",
    "eval_runs": "Scored task outcomes used as the RSI fitness signal.",
    "memories": "Long-horizon notes the harness may load as context.",
    "query_traces": "Audit log of NL → mongosh generations.",
    "agent_events": "Append-only event log for long-horizon work.",
}
