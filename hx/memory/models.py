"""Memory documents. Same shapes in Atlas and in the local fallback store."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

RULE_MAX_CHARS = 300
RULE_CAP = 12          # trial + active rules; the harness must stay small
CASE_DESC_MAX = 160


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:max_len].rstrip("-")) or new_id()


# ---------------------------------------------------------------- general harness
class Rule(BaseModel):
    """A standing rule, loaded every turn (like a line in CLAUDE.md)."""
    rule_id: str
    text: str
    why: str = ""
    status: Literal["trial", "active", "retired"] = "trial"
    trials: int = 0            # turns where the review judged it relevant
    helped: int = 0
    strikes: int = 0           # relevant turns where the failure it targets came back anyway
    source_trace: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class HarnessVersion(BaseModel):
    """Immutable-by-convention snapshot of the general rules. Lineage = parent chain."""
    version: int
    parent: Optional[int] = None
    rules: list[Rule] = Field(default_factory=list)
    change: str = "init"       # e.g. "add rule no-interactive-run (trial)"
    reason: str = ""
    trace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=now)

    def live_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.status != "retired"]

    def get(self, rule_id: str) -> Optional[Rule]:
        return next((r for r in self.rules if r.rule_id == rule_id), None)


# ---------------------------------------------------------------- case library
class Case(BaseModel):
    """One incident. Only its one-line description is in context until recall selects it."""
    case_id: str
    title: str
    description: str           # the hook recall reads; be specific
    body: str                  # markdown: Symptom / Root cause / When it applies / How to apply
    tags: list[str] = Field(default_factory=list)
    status: Literal["active", "retired"] = "active"
    uses: int = 0              # times loaded into a turn
    helped: int = 0
    misleading: int = 0
    source_trace: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------- traces
class ToolEvent(BaseModel):
    step: int
    tool: str
    args: str                  # one-line summary, never full file contents
    ok: bool
    kind: str = "ok"
    output: str = ""           # short excerpt
    benign: bool = True


class TurnTrace(BaseModel):
    trace_id: str = Field(default_factory=new_id)
    session_id: str = ""
    created_at: datetime = Field(default_factory=now)
    task: str
    workdir: str = ""
    harness_version: int = 0
    rule_ids: list[str] = Field(default_factory=list)
    loaded_case_ids: list[str] = Field(default_factory=list)
    events: list[ToolEvent] = Field(default_factory=list)
    final_text: str = ""
    steps: int = 0
    outcome: Literal["ok", "recovered", "failed", "aborted"] = "ok"
    abort_reason: Optional[str] = None   # step_limit | interrupted | model_error | malformed_output
    user_feedback: Optional[str] = None
    system_tokens: int = 0     # size of the assembled harness context
    prompt_tokens: int = 0
    completion_tokens: int = 0
    memory_actions: list[str] = Field(default_factory=list)

    def failures(self) -> list[ToolEvent]:
        return [e for e in self.events if not e.ok and not e.benign]
