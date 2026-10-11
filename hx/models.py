from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Policy(BaseModel):
    """Knobs C may mutate. apply_edits(set_policy) must change only the named field."""

    max_retries: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024
    include_schema: bool = False
    include_samples: bool = False
    include_indexes: bool = False


class ContextComponent(BaseModel):
    id: str
    title: str
    text: str
    enabled: bool = True


class Harness(BaseModel):
    harness_id: str
    version: int = 1
    parent_version: int | None = None
    rules: list[str] = Field(default_factory=list)
    policy: Policy = Field(default_factory=Policy)
    components: list[ContextComponent] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    mutation_reason: str = ""


class Edit(BaseModel):
    op: Literal["delete_rule", "add_rule", "set_policy", "set_component_enabled"]
    value: Any = None
    knob: str | None = None
    component_id: str | None = None


class AssembledPrompt(BaseModel):
    system: str
    user: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    context_components: list[str]


class LLMResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    fake: bool = False
    retries: int = 0


class Task(BaseModel):
    task_id: str
    nl: str
    split: Literal["train", "heldout"] = "train"
    database: str = "sample_mflix"
    collection: str = "movies"
    gold: Any | None = None


class Trace(BaseModel):
    harness_id: str
    harness_version: int
    task_id: str
    seed: int = 0
    split: Literal["train", "heldout"] = "train"
    natural_language: str
    context_components: list[str] = Field(default_factory=list)
    raw_response: str = ""
    parsed: dict[str, Any] | None = None
    pipeline: list[dict[str, Any]] | None = None
    retries: int = 0
    score: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    executor_result: Any | None = None


class GenerationReport(BaseModel):
    harness_id: str
    seed: int
    train_scores: dict[str, float | None] = Field(default_factory=dict)
    heldout_scores: dict[str, float | None] = Field(default_factory=dict)
    tokens_per_task: dict[str, int] = Field(default_factory=dict)
    traces: list[Trace] = Field(default_factory=list)
    elapsed_s: float = 0.0
    cached: bool = False
