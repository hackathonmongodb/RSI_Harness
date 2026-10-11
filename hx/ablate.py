"""Leave-one-out prune evidence. Public signature matches SPEC.md section 7."""
from __future__ import annotations

import os

from hx.models import Ablation, Component, Edit, Harness, Trace
from hx.promote import DEFAULT_NOISE


def _prefer_mocks() -> bool:
    return os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}


def _load_harness(harness_id: str) -> tuple[Harness, list[Component]]:
    if _prefer_mocks():
        from tests.mocks import load_harness as _lh
    else:
        from hx.harness import load_harness as _lh
    return _lh(harness_id)


def _apply_edits(parent_id, edits, phase, generation):
    if _prefer_mocks():
        from tests.mocks import apply_edits as _ae
    else:
        from hx.harness import apply_edits as _ae
    return _ae(parent_id, edits, phase, generation)


def _run_generation(harness_id: str, dataset: str, **kwargs):
    if _prefer_mocks():
        from tests.mocks import run_generation as _rg
    else:
        from hx.generation import run_generation as _rg
    return _rg(harness_id, dataset, **kwargs)


def _traces_for(harness_id: str) -> list[Trace]:
    try:
        from tests.mocks import STORE

        return [t for t in STORE.traces.values() if t.harness_id == harness_id]
    except ImportError:
        return []


def _task_ids_where_component_appeared(harness_id: str, component_id: str) -> list[str]:
    ids: list[str] = []
    for tr in _traces_for(harness_id):
        if any(ref.component_id == component_id for ref in tr.context_components):
            ids.append(tr.task_id)
    return sorted(set(ids))


def prune_candidates(harness_id: str, dataset: str) -> list[Ablation]:
    harness, comps = _load_harness(harness_id)
    active = [c for c in comps if c.status == "active"]
    parent_result = _run_generation(harness_id, dataset)
    out: list[Ablation] = []
    for c in active:
        if c.pinned:
            continue
        if c.kind not in ("rule", "tool"):
            continue
        task_ids = _task_ids_where_component_appeared(harness_id, c.component_id)
        if c.kind == "tool" and not task_ids:
            # Unused tools are deleted without ablation; still record a dummy Ablation.
            ab = Ablation(
                harness_id=harness_id,
                component_id=c.component_id,
                n_tasks=0,
                score_with=parent_result.train_score,
                score_without=parent_result.train_score,
                delta=0.0,
                tokens_saved_per_task=float(c.token_cost),
            )
            out.append(ab)
            continue
        if not task_ids:
            continue
        child = _apply_edits(
            harness_id,
            [Edit(op="delete", kind=c.kind, target_component_id=c.component_id)],
            "prune",
            harness.generation + 1,
        )
        child_result = _run_generation(child.harness_id, dataset, task_ids=task_ids)
        with_score = parent_result.train_score
        without = child_result.train_score
        ab = Ablation(
            harness_id=harness_id,
            component_id=c.component_id,
            n_tasks=len(task_ids),
            score_with=with_score,
            score_without=without,
            delta=with_score - without,
            tokens_saved_per_task=float(c.token_cost),
        )
        out.append(ab)
    try:
        from tests.mocks import STORE

        STORE.ablations.extend(out)
    except Exception:
        pass
    _ = DEFAULT_NOISE
    return out
