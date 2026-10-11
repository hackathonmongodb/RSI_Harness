"""hx/models.py — shared data contracts. Source of truth for every collection.

Place at hx/models.py. Everyone imports from here. Changing a field means: edit this file,
update SPEC.md, and tell the team in the same commit.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["rule", "policy", "tool"]
Op = Literal["add", "modify", "delete"]
Phase = Literal["baseline", "policy", "growth", "prune"]
Split = Literal["train", "heldout"]
ErrorType = Literal["none", "parse", "blocked", "exec_error", "timeout", "wrong_result"]


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- hx_eval (admin only)
class Task(BaseModel):
    """hx_eval.tasks. Gold is stored separately in hx_eval.gold, never on this doc."""
    task_id: str
    dataset: str                      # "sample_mflix" | "sample_analytics" | ...
    split: Split
    question: str                     # MUST name the exact output fields, plus any ordering / limit
    collection: str                   # primary collection the pipeline runs against
    ordered: bool = False             # True if row order matters for scoring
    tags: list[str] = Field(default_factory=list)   # e.g. decimal128, date, lookup, array, nested, group


class Gold(BaseModel):
    """hx_eval.gold"""
    task_id: str
    pipeline: list[dict[str, Any]]    # EJSON-compatible
    verified: bool = False            # hand-checked by a human


class HeldoutResult(BaseModel):
    """hx_eval.heldout_results — scores only, no outputs. Evolve agent can never read this DB."""
    harness_id: str
    task_id: str
    seed: int = 0
    score: float
    total_tokens: int = 0


# ---------------------------------------------------------------- hx_harness (meta: readWrite)
class Component(BaseModel):
    """hx_harness.components — one doc per VERSION of a component."""
    component_id: str                 # stable logical id, e.g. "rule.decimal128", "policy.cot"
    version: int = 1
    kind: Kind
    name: str
    content: Optional[str] = None     # rule text (<= 400 chars, enforced by $jsonSchema)
    value: Optional[Any] = None       # policy value
    template: Optional[dict] = None   # tool: {"description", "params": {name: type}, "pipeline": [... "{{param}}" ...]}
    pinned: bool = False              # pinned components can never be edited or deleted by the meta-agent
    dataset_scope: str = "*"
    status: Literal["active", "retired", "rejected"] = "active"
    token_cost: int = 0               # approx tokens this component adds to the prompt
    created_gen: int = 0
    created_by_edit: Optional[str] = None


class ComponentRef(BaseModel):
    component_id: str
    version: int


class Harness(BaseModel):
    """hx_harness.harnesses — a snapshot: set of component versions. Lineage = parent_id chain."""
    harness_id: str
    parent_id: Optional[str] = None
    dataset: str
    generation: int
    phase: Phase
    components: list[ComponentRef]
    train_score: Optional[float] = None
    heldout_score: Optional[float] = None
    tokens_per_task: Optional[float] = None
    n_rules: int = 0
    n_tools: int = 0
    status: Literal["candidate", "promoted", "rejected", "static"] = "candidate"   # "static" = reference baselines
    created_at: datetime = Field(default_factory=now)


class Edit(BaseModel):
    op: Op
    kind: Kind
    target_component_id: str
    component: Optional[Component] = None   # new/modified version for add/modify; None for delete


class Manifest(BaseModel):
    """hx_harness.manifests — one per edit. Falsifiable: predictions are scored next round."""
    edit_id: str
    generation: int
    phase: Phase
    parent_harness_id: str
    child_harness_id: Optional[str] = None
    edit: Edit
    evidence_trace_ids: list[str] = Field(default_factory=list)   # train traces only
    root_cause: str
    targeted_fix: str
    predicted_fixes: list[str] = Field(default_factory=list)       # train task_ids expected to flip fail -> pass
    predicted_regressions: list[str] = Field(default_factory=list) # train task_ids at risk of pass -> fail
    actual_fixed: list[str] = Field(default_factory=list)
    actual_regressed: list[str] = Field(default_factory=list)
    gate_reason: Optional[str] = None
    verdict: Optional[Literal["promoted", "rejected"]] = None
    created_at: datetime = Field(default_factory=now)


class Ablation(BaseModel):
    """hx_harness.ablations — leave-one-out results used as evidence for delete edits."""
    harness_id: str
    component_id: str
    n_tasks: int
    score_with: float
    score_without: float
    delta: float                      # score_with - score_without; ~0 or negative => dead weight
    tokens_saved_per_task: float = 0.0


# ---------------------------------------------------------------- hx_evidence (meta: read)
class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result_tokens: int = 0


class Trace(BaseModel):
    """hx_evidence.traces — TRAIN split only. Held-out runs write HeldoutResult instead."""
    trace_id: str
    harness_id: str
    generation: int
    task_id: str
    seed: int = 0
    context_components: list[ComponentRef] = Field(default_factory=list)   # what was in the prompt
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_output: str = ""
    parsed: Optional[dict] = None     # {"collection": str, "pipeline": [...]}  (tool calls already expanded)
    attempts: int = 1
    error: Optional[str] = None
    error_type: ErrorType = "none"
    score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class Report(BaseModel):
    """hx_evidence.reports — debugger output, one per analyzed trace."""
    trace_id: str
    task_id: str
    harness_id: str
    outcome: Literal["fail", "pass"]
    root_cause: str
    implicated_component_id: Optional[str] = None   # None + missing_component=True => something should be ADDED
    implicated_kind: Optional[Kind] = None
    missing_component: bool = False
    confidence: float = 0.5
    evidence: str = ""                # short excerpt from the trace


# ---------------------------------------------------------------- orchestrator return types
class GenerationResult(BaseModel):
    harness_id: str
    train_score: float
    heldout_score: float
    tokens_per_task: float
    train_outcomes: dict[str, float]      # task_id -> score. Safe to show the evolve agent.
    heldout_outcomes: dict[str, float]    # ORCHESTRATOR ONLY. Never pass to any LLM.
    failed_train_trace_ids: list[str]
    passed_train_trace_ids: list[str]
