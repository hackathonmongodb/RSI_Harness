"""Meta-agent edits. Public signature matches SPEC.md section 7.

Caps, pinned components, policy ranges, evidence, and predicted task ids are
enforced in code (retry LLM once, then drop the edit).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional
from uuid import uuid4

from hx.models import (
    Component,
    Edit,
    Harness,
    Kind,
    Manifest,
    Phase,
    Trace,
)

POLICY_VALUES: dict[str, Any] = {
    "policy.n_sample_docs": frozenset({0, 1, 2, 3}),
    "policy.doc_truncate_chars": None,  # 100..800 inclusive, checked separately
    "policy.schema_style": frozenset({"names", "ts_annotated"}),
    "policy.include_indexes": frozenset({False, True, 0, 1}),
    "policy.cot": frozenset({False, True, 0, 1}),
    "policy.max_retries": frozenset({0, 1, 2}),
    "policy.sample_docs_tool": frozenset({False, True, 0, 1}),
}

BOOL_POLICIES = {
    "policy.include_indexes",
    "policy.cot",
    "policy.sample_docs_tool",
}

PHASE_CAPS = {"policy": 2, "growth": 3, "prune": 3, "baseline": 0}
RULE_CONTENT_MAX = 400
SOFT_RULE_CAP = 12
PINNED_IDS = {"rule.base"}


def _prefer_mocks() -> bool:
    return os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}


def _complete(model: str, system: str, messages: list[dict], **kwargs):
    if _prefer_mocks():
        from tests.mocks import complete as _c
        return _c(model, system, messages, **kwargs)
    from hx.llm import complete as _c
    return _c(model, system, messages, **kwargs)


def _load_harness(harness_id: str) -> tuple[Harness, list[Component]]:
    if _prefer_mocks():
        from tests.mocks import load_harness as _lh
    else:
        from hx.harness import load_harness as _lh
    return _lh(harness_id)


def fetch_trace(trace_id: str) -> Optional[Trace]:
    """Train traces only. Never joins gold or held-out results."""
    try:
        from tests.mocks import STORE

        tr = STORE.traces.get(trace_id)
        if tr is None:
            return None
        return tr
    except ImportError:
        return None


def fetch_report(trace_id: str) -> Optional[dict[str, Any]]:
    try:
        from tests.mocks import STORE

        for r in STORE.reports:
            if r.get("trace_id") == trace_id:
                return r
        return None
    except ImportError:
        return None


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def _policy_value_ok(component_id: str, value: Any) -> bool:
    if component_id not in POLICY_VALUES:
        return False
    value = _coerce_bool(value) if component_id in BOOL_POLICIES else value
    if component_id == "policy.doc_truncate_chars":
        try:
            n = int(value)
        except (TypeError, ValueError):
            return False
        return 100 <= n <= 800
    allowed = POLICY_VALUES[component_id]
    if component_id in BOOL_POLICIES:
        value = _coerce_bool(value)
        if value in (0, 1):
            value = bool(value)
        return value in {False, True}
    if component_id == "policy.n_sample_docs" or component_id == "policy.max_retries":
        try:
            value = int(value)
        except (TypeError, ValueError):
            return False
    return value in allowed


def validate_edit(
    edit: Edit,
    phase: Phase,
    comps: list[Component],
    n_active_rules: int,
) -> Optional[str]:
    """Return an error string, or None if the edit is allowed."""
    by_id = {c.component_id: c for c in comps}
    target = edit.target_component_id
    existing = by_id.get(target)

    if target in PINNED_IDS or (existing is not None and existing.pinned):
        return f"pinned component cannot be edited: {target}"

    if phase == "policy":
        if edit.kind != "policy" or edit.op != "modify":
            return "policy phase allows modify of existing policy keys only"
        if target not in POLICY_VALUES or existing is None:
            return f"unknown or new policy key: {target}"
        if edit.component is None:
            return "policy modify requires component.value"
        if not _policy_value_ok(target, edit.component.value):
            return f"policy value out of range: {target}={edit.component.value!r}"

    if phase == "growth" and edit.kind == "policy":
        return "growth phase is for rules/tools, not policies"

    if edit.kind == "policy" and edit.op != "modify":
        return "policies may only be modified, never added or deleted"

    if edit.kind == "policy" and target not in POLICY_VALUES:
        return f"policy key not in SPEC table: {target}"

    if edit.kind == "policy" and edit.component is not None:
        if not _policy_value_ok(target, edit.component.value):
            return f"policy value out of range: {target}={edit.component.value!r}"

    if edit.op == "modify" and existing is None and edit.kind != "rule":
        return f"modify of missing component: {target}"

    if edit.kind == "rule" and edit.op in ("add", "modify") and edit.component is not None:
        content = edit.component.content or ""
        if len(content) > RULE_CONTENT_MAX:
            return f"rule content exceeds {RULE_CONTENT_MAX} chars"

    if (
        phase == "growth"
        and edit.op == "add"
        and edit.kind == "rule"
        and n_active_rules >= SOFT_RULE_CAP
    ):
        return "soft cap of 12 active rules; add requires a paired delete"

    if phase == "prune" and edit.op != "delete":
        return "prune phase allows delete only"

    if phase == "growth" and edit.op not in ("add", "modify"):
        return "growth phase allows add or modify only"

    return None


def validate_manifest(
    manifest: Manifest,
    phase: Phase,
    comps: list[Component],
    failed_train_task_ids: set[str],
    passed_train_task_ids: set[str],
    n_active_rules: int,
) -> Optional[str]:
    if not (manifest.root_cause or "").strip():
        return "empty root_cause"
    if not (manifest.targeted_fix or "").strip():
        return "empty targeted_fix"
    if phase != "prune" and not manifest.evidence_trace_ids:
        return "no evidence, no edit"
    err = validate_edit(manifest.edit, phase, comps, n_active_rules)
    if err:
        return err
    invented_fix = [t for t in manifest.predicted_fixes if t not in failed_train_task_ids]
    if invented_fix:
        return f"predicted_fixes not in train failures: {invented_fix}"
    invented_reg = [t for t in manifest.predicted_regressions if t not in passed_train_task_ids]
    if invented_reg:
        return f"predicted_regressions not in train passes: {invented_reg}"
    return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return m.group(1) if m else text


def _parse_edits_payload(text: str) -> list[dict[str, Any]]:
    raw = json.loads(_strip_fences(text))
    if isinstance(raw, dict) and "edits" in raw:
        return list(raw["edits"])
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    return []


def _edit_from_dict(d: dict[str, Any], comps: list[Component]) -> Edit:
    op = d["op"]
    kind: Kind = d.get("kind") or "rule"
    target = d["target_component_id"]
    component = None
    if op in ("add", "modify"):
        by_id = {c.component_id: c for c in comps}
        prev = by_id.get(target)
        if kind == "policy":
            component = Component(
                component_id=target,
                kind="policy",
                name=target.split(".", 1)[-1],
                value=_coerce_bool(d.get("value", d.get("component", {}).get("value") if isinstance(d.get("component"), dict) else None)),
            )
        elif kind == "rule":
            content = d.get("content")
            if content is None and isinstance(d.get("component"), dict):
                content = d["component"].get("content")
            component = Component(
                component_id=target,
                kind="rule",
                name=target.split(".", 1)[-1],
                content=content or "",
                pinned=False,
            )
        elif kind == "tool" and prev is not None:
            component = prev
    return Edit(op=op, kind=kind, target_component_id=target, component=component)


def _passing_train_outputs(passed_trace_ids: list[str]) -> list[dict[str, Any]]:
    """Few-shot material from passing train traces' own outputs — never gold."""
    shots = []
    for tid in passed_trace_ids:
        tr = fetch_trace(tid)
        if tr is None or tr.score < 1.0:
            continue
        shots.append({"task_id": tr.task_id, "output": tr.raw_output or tr.parsed})
    return shots


def _meta_model() -> str:
    import os

    return os.environ.get(
        "META_MODEL",
        os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


def propose_edits(harness_id: str, phase: Phase, overview: dict) -> list[Manifest]:
    harness, comps = _load_harness(harness_id)
    failed_ids = set(overview.get("failed_train_task_ids") or [])
    passed_ids = set(overview.get("passed_train_task_ids") or [])
    passed_trace_ids = list(overview.get("passed_train_trace_ids") or [])
    n_rules = sum(1 for c in comps if c.kind == "rule" and c.status == "active")

    if phase == "prune":
        return _prune_manifests(harness_id, harness, comps, overview, failed_ids, passed_ids, n_rules)

    shots = _passing_train_outputs(passed_trace_ids)
    system = (
        "You propose harness edits as JSON. No gold pipelines. No held-out tasks. "
        "Output {\"edits\":[{op, kind, target_component_id, value?, content?, "
        "root_cause, targeted_fix, evidence_trace_ids, predicted_fixes, predicted_regressions}]}. "
        "Few-shot rules must copy passing train outputs, never gold."
    )
    user = {
        "overview": {k: v for k, v in overview.items() if k != "heldout_outcomes"},
        "phase": phase,
        "component_ids": [
            {"component_id": c.component_id, "kind": c.kind, "pinned": c.pinned, "content": c.content, "value": c.value}
            for c in comps
        ],
        "passing_train_outputs_for_fewshot": shots,
        "failed_train_task_ids": sorted(failed_ids),
        "passed_train_task_ids": sorted(passed_ids),
    }
    messages = [{"role": "user", "content": json.dumps(user)}]
    text = _complete(_meta_model(), system, messages).text
    manifests = _materialize(text, harness_id, phase, harness, comps, failed_ids, passed_ids, n_rules)
    if not manifests:
        text = _complete(_meta_model(), system + " Retry: last output failed validation.", messages).text
        manifests = _materialize(text, harness_id, phase, harness, comps, failed_ids, passed_ids, n_rules)
    cap = PHASE_CAPS.get(phase, 0)
    return manifests[:cap]


def _materialize(
    text: str,
    harness_id: str,
    phase: Phase,
    harness: Harness,
    comps: list[Component],
    failed_ids: set[str],
    passed_ids: set[str],
    n_rules: int,
) -> list[Manifest]:
    try:
        rows = _parse_edits_payload(text)
    except (json.JSONDecodeError, TypeError, KeyError):
        return []
    out: list[Manifest] = []
    for d in rows:
        try:
            edit = _edit_from_dict(d, comps)
            m = Manifest(
                edit_id=uuid4().hex[:12],
                generation=harness.generation + 1,
                phase=phase,
                parent_harness_id=harness_id,
                edit=edit,
                evidence_trace_ids=list(d.get("evidence_trace_ids") or []),
                root_cause=str(d.get("root_cause") or ""),
                targeted_fix=str(d.get("targeted_fix") or ""),
                predicted_fixes=list(d.get("predicted_fixes") or []),
                predicted_regressions=list(d.get("predicted_regressions") or []),
            )
        except Exception:
            continue
        err = validate_manifest(m, phase, comps, failed_ids, passed_ids, n_rules)
        if err:
            continue
        out.append(m)
    return out


def _prune_manifests(
    harness_id: str,
    harness: Harness,
    comps: list[Component],
    overview: dict,
    failed_ids: set[str],
    passed_ids: set[str],
    n_rules: int,
) -> list[Manifest]:
    """Deterministic deletes from ablation docs — not META_MODEL."""
    try:
        from tests.mocks import STORE

        ablations = [a for a in STORE.ablations if getattr(a, "harness_id", None) == harness_id]
    except ImportError:
        ablations = list(overview.get("ablations") or [])
    noise = float(overview.get("noise", 0.05))
    ranked = sorted(
        [a for a in ablations if getattr(a, "delta", 1) <= noise],
        key=lambda a: getattr(a, "delta", 0),
    )
    out: list[Manifest] = []
    for a in ranked[: PHASE_CAPS["prune"]]:
        cid = a.component_id
        edit = Edit(op="delete", kind="rule", target_component_id=cid)
        kind = next((c.kind for c in comps if c.component_id == cid), "rule")
        edit = Edit(op="delete", kind=kind, target_component_id=cid)
        m = Manifest(
            edit_id=uuid4().hex[:12],
            generation=harness.generation + 1,
            phase="prune",
            parent_harness_id=harness_id,
            edit=edit,
            evidence_trace_ids=list(overview.get("evidence_trace_ids") or []),
            root_cause=f"ablation delta {a.delta} <= noise for {cid}",
            targeted_fix=f"delete {cid}",
            predicted_fixes=[],
            predicted_regressions=[],
        )
        err = validate_manifest(m, "prune", comps, failed_ids, passed_ids, n_rules)
        if err:
            continue
        out.append(m)
    return out
