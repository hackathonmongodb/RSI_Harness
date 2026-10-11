"""In-memory fakes matching SPEC.md section 7 signatures exactly.

Used until A/B land. Do not change argument names, defaults, or return types.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hx.models import (
    Component,
    ComponentRef,
    Edit,
    ErrorType,
    GenerationResult,
    Harness,
    Manifest,
    Phase,
    Task,
    Trace,
)

# ---------------------------------------------------------------------------
# Optional complete() — identical to SPEC §7 hx/llm.py
# ---------------------------------------------------------------------------


class LLMResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class Store:
    def __init__(self) -> None:
        self.components: dict[tuple[str, int], Component] = {}
        self.harnesses: dict[str, Harness] = {}
        self.tasks: dict[str, Task] = {}
        self.traces: dict[str, Trace] = {}
        self.manifests: list[Manifest] = []
        self.ablations: list[Any] = []
        self.reports: list[dict[str, Any]] = []
        self.task_scores: dict[str, dict[str, float]] = {}  # harness_id -> task_id -> score
        self.llm_oracle: dict[str, str] = {}  # substring in prompt -> canned JSON text
        self.last_complete_payloads: list[dict[str, Any]] = []
        self.reset_baseline()

    def reset_baseline(self) -> None:
        self.components.clear()
        self.harnesses.clear()
        self.tasks.clear()
        self.traces.clear()
        self.manifests.clear()
        self.ablations.clear()
        self.reports.clear()
        self.task_scores.clear()
        self.llm_oracle.clear()
        self.last_complete_payloads.clear()

        base = Component(
            component_id="rule.base",
            version=1,
            kind="rule",
            name="base",
            content="Return JSON {collection, pipeline} in Extended JSON. No $out/$merge.",
            pinned=True,
            status="active",
        )
        policies = [
            Component(component_id="policy.n_sample_docs", version=1, kind="policy", name="n_sample_docs", value=0),
            Component(component_id="policy.doc_truncate_chars", version=1, kind="policy", name="doc_truncate_chars", value=300),
            Component(component_id="policy.schema_style", version=1, kind="policy", name="schema_style", value="names"),
            Component(component_id="policy.include_indexes", version=1, kind="policy", name="include_indexes", value=False),
            Component(component_id="policy.cot", version=1, kind="policy", name="cot", value=False),
            Component(component_id="policy.max_retries", version=1, kind="policy", name="max_retries", value=0),
            Component(component_id="policy.sample_docs_tool", version=1, kind="policy", name="sample_docs_tool", value=False),
        ]
        tip = Component(
            component_id="rule.types",
            version=1,
            kind="rule",
            name="types",
            content="Use Decimal128 ranges, not exact equality.",
            pinned=False,
            status="active",
            token_cost=10,
        )
        unused_tool = Component(
            component_id="tool.unused",
            version=1,
            kind="tool",
            name="unused",
            template={"description": "unused", "params": {}, "pipeline": []},
            pinned=False,
            status="active",
        )
        for c in (base, *policies, tip, unused_tool):
            self.components[(c.component_id, c.version)] = c

        refs = [
            ComponentRef(component_id=c.component_id, version=c.version)
            for c in (base, *policies, tip, unused_tool)
        ]
        h0 = Harness(
            harness_id="h0",
            parent_id=None,
            dataset="sample_mflix",
            generation=0,
            phase="baseline",
            components=refs,
            n_rules=2,
            n_tools=1,
            status="static",
            train_score=0.4,
            heldout_score=0.4,
            tokens_per_task=100.0,
        )
        self.harnesses["h0"] = h0

        fail_ids = [f"train_fail_{i}" for i in range(3)]
        pass_ids = [f"train_pass_{i}" for i in range(2)]
        held_ids = [f"held_{i}" for i in range(2)]
        for tid in fail_ids:
            self.tasks[tid] = Task(
                task_id=tid,
                dataset="sample_mflix",
                split="train",
                question=f"List title and year for {tid}. Output fields: title, year.",
                collection="movies",
                tags=["date"],
            )
        for tid in pass_ids:
            self.tasks[tid] = Task(
                task_id=tid,
                dataset="sample_mflix",
                split="train",
                question=f"Count movies for {tid}. Output fields: n.",
                collection="movies",
                tags=["group"],
            )
        for tid in held_ids:
            self.tasks[tid] = Task(
                task_id=tid,
                dataset="sample_mflix",
                split="heldout",
                question="HELD OUT — must never appear in a meta prompt",
                collection="movies",
            )

        ctx = [
            ComponentRef(component_id="rule.base", version=1),
            ComponentRef(component_id="rule.types", version=1),
            ComponentRef(component_id="policy.cot", version=1),
        ]
        for tid in fail_ids:
            tr = Trace(
                trace_id=f"tr_{tid}",
                harness_id="h0",
                generation=0,
                task_id=tid,
                context_components=ctx,
                raw_output='{"collection":"movies","pipeline":[]}',
                parsed={"collection": "movies", "pipeline": []},
                error="wrong_result: extra field",
                error_type="wrong_result",
                score=0.0,
            )
            self.traces[tr.trace_id] = tr
        for tid in pass_ids:
            tr = Trace(
                trace_id=f"tr_{tid}",
                harness_id="h0",
                generation=0,
                task_id=tid,
                context_components=ctx,
                raw_output='{"collection":"movies","pipeline":[{"$count":"n"}]}',
                parsed={"collection": "movies", "pipeline": [{"$count": "n"}]},
                error_type="none",
                score=1.0,
            )
            self.traces[tr.trace_id] = tr

        scores = {tid: 0.0 for tid in fail_ids}
        scores.update({tid: 1.0 for tid in pass_ids})
        self.task_scores["h0"] = scores


STORE = Store()


def reset_store() -> None:
    STORE.reset_baseline()


# ---------------------------------------------------------------------------
# SPEC §7 fakes
# ---------------------------------------------------------------------------


def score(task: Task, parsed: dict | None) -> tuple[float, ErrorType, str | None]:
    if parsed is None:
        return 0.0, "parse", "not valid JSON / EJSON"
    if not isinstance(parsed, dict) or "collection" not in parsed or "pipeline" not in parsed:
        return 0.0, "parse", "missing collection or pipeline"
    if not isinstance(parsed.get("pipeline"), list):
        return 0.0, "blocked", "pipeline must be a list"
    outcome = STORE.task_scores.get("h0", {}).get(task.task_id)
    if outcome == 1.0:
        return 1.0, "none", None
    return 0.0, "wrong_result", "result mismatch"


def get_noise_floor(dataset: str) -> float:
    return 0.05


def load_harness(harness_id: str) -> tuple[Harness, list[Component]]:
    h = STORE.harnesses[harness_id]
    comps = [
        STORE.components[(ref.component_id, ref.version)]
        for ref in h.components
        if (ref.component_id, ref.version) in STORE.components
    ]
    return h, comps


def apply_edits(parent_id: str, edits: list[Edit], phase: Phase, generation: int) -> Harness:
    parent, comps = load_harness(parent_id)
    by_id = {c.component_id: c for c in comps}
    for ed in edits:
        if ed.op == "delete":
            by_id.pop(ed.target_component_id, None)
        elif ed.op in ("add", "modify"):
            new_c = ed.component
            if new_c is None:
                existing = by_id.get(ed.target_component_id)
                if existing is None:
                    continue
                new_c = existing.model_copy(update={"version": existing.version + 1})
            else:
                prev = by_id.get(ed.target_component_id)
                ver = (prev.version + 1) if prev is not None else 1
                new_c = new_c.model_copy(update={"version": ver, "component_id": ed.target_component_id})
            STORE.components[(new_c.component_id, new_c.version)] = new_c
            by_id[new_c.component_id] = new_c
    child_id = f"{parent_id}_{phase}_{generation}_{uuid4().hex[:8]}"
    refs = [ComponentRef(component_id=c.component_id, version=c.version) for c in by_id.values()]
    n_rules = sum(1 for c in by_id.values() if c.kind == "rule")
    n_tools = sum(1 for c in by_id.values() if c.kind == "tool")
    child = Harness(
        harness_id=child_id,
        parent_id=parent_id,
        dataset=parent.dataset,
        generation=generation,
        phase=phase,
        components=refs,
        n_rules=n_rules,
        n_tools=n_tools,
        status="candidate",
    )
    STORE.harnesses[child_id] = child
    parent_scores = STORE.task_scores.get(parent_id, {})
    STORE.task_scores[child_id] = dict(parent_scores)
    return child


def run_generation(
    harness_id: str,
    dataset: str,
    seeds: list[int] = [0],
    parallel: int = 8,
    task_ids: list[str] | None = None,
) -> GenerationResult:
    train_tasks = [t for t in STORE.tasks.values() if t.split == "train" and t.dataset == dataset]
    held_tasks = [t for t in STORE.tasks.values() if t.split == "heldout" and t.dataset == dataset]
    if task_ids is not None:
        allow = set(task_ids)
        train_tasks = [t for t in train_tasks if t.task_id in allow]
        held_tasks = []

    scores_map = STORE.task_scores.get(harness_id, {})
    train_outcomes = {t.task_id: float(scores_map.get(t.task_id, 0.0)) for t in train_tasks}
    heldout_outcomes = {t.task_id: float(scores_map.get(t.task_id, 0.4)) for t in held_tasks}

    failed_ids: list[str] = []
    passed_ids: list[str] = []
    for t in train_tasks:
        tr_id = f"tr_{t.task_id}"
        if tr_id not in STORE.traces:
            STORE.traces[tr_id] = Trace(
                trace_id=tr_id,
                harness_id=harness_id,
                generation=STORE.harnesses.get(harness_id, Harness(
                    harness_id=harness_id, dataset=dataset, generation=0, phase="baseline", components=[]
                )).generation,
                task_id=t.task_id,
                score=train_outcomes[t.task_id],
                error_type="wrong_result" if train_outcomes[t.task_id] < 1 else "none",
            )
        else:
            STORE.traces[tr_id] = STORE.traces[tr_id].model_copy(update={"harness_id": harness_id})
        if train_outcomes[t.task_id] >= 1.0:
            passed_ids.append(tr_id)
        else:
            failed_ids.append(tr_id)

    n_train = max(len(train_outcomes), 1)
    train_score = sum(train_outcomes.values()) / n_train
    n_held = len(heldout_outcomes)
    heldout_score = (sum(heldout_outcomes.values()) / n_held) if n_held else train_score
    h = STORE.harnesses.get(harness_id)
    tokens = float(h.tokens_per_task) if h and h.tokens_per_task is not None else 100.0
    if h is not None:
        STORE.harnesses[harness_id] = h.model_copy(
            update={"train_score": train_score, "heldout_score": heldout_score, "tokens_per_task": tokens}
        )
    return GenerationResult(
        harness_id=harness_id,
        train_score=train_score,
        heldout_score=heldout_score,
        tokens_per_task=tokens,
        train_outcomes=train_outcomes,
        heldout_outcomes=heldout_outcomes,
        failed_train_trace_ids=failed_ids,
        passed_train_trace_ids=passed_ids,
    )


def complete(
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> LLMResult:
    blob = system + "\n" + json.dumps(messages)
    STORE.last_complete_payloads.append(
        {
            "model": model,
            "system": system,
            "messages": copy.deepcopy(messages),
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )
    for needle, text in STORE.llm_oracle.items():
        if needle in blob:
            return LLMResult(text=text, prompt_tokens=len(blob) // 4, completion_tokens=len(text) // 4)
    if "implicated_component_id" in system or "debugger" in system.lower():
        text = json.dumps(
            {
                "root_cause": "type mismatch",
                "implicated_component_id": "rule.types",
                "implicated_kind": "rule",
                "missing_component": False,
                "confidence": 0.8,
                "evidence": "wrong_result",
            }
        )
        return LLMResult(text=text, prompt_tokens=10, completion_tokens=10)
    text = json.dumps(
        {
            "edits": [
                {
                    "op": "modify",
                    "kind": "policy",
                    "target_component_id": "policy.cot",
                    "value": True,
                    "root_cause": "parse failures without reasoning",
                    "targeted_fix": "turn CoT on",
                    "evidence_trace_ids": ["tr_train_fail_0"],
                    "predicted_fixes": ["train_fail_0"],
                    "predicted_regressions": ["train_pass_0"],
                }
            ]
        }
    )
    return LLMResult(text=text, prompt_tokens=10, completion_tokens=10)


def get_traces(harness_id: str, trace_ids: list[str]) -> list[Trace]:
    out: list[Trace] = []
    for tid in trace_ids:
        tr = STORE.traces.get(tid)
        if tr is None:
            continue
        if harness_id and tr.harness_id != harness_id:
            continue
        out.append(tr)
    return out


def get_task(task_id: str) -> Optional[Task]:
    return STORE.tasks.get(task_id)
