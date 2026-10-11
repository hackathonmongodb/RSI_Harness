"""Evolution loop. Public signature matches SPEC.md section 7."""
from __future__ import annotations

import logging
import os
import random
from typing import Any

from hx.analytics import prediction_scores, set_manifests
from hx.models import GenerationResult, Manifest, Phase, Report
from hx.promote import DEFAULT_NOISE, gate, merge_promoted

log = logging.getLogger("hx.loop")

CADENCE: tuple[Phase, ...] = ("policy", "growth", "growth", "prune")


def _cadence(generation: int) -> Phase:
    """generation 1..N -> policy, growth, growth, prune repeating."""
    if generation < 1:
        return "baseline"
    return CADENCE[(generation - 1) % 4]


def _prefer_mocks() -> bool:
    return os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}


def _run_generation(harness_id: str, dataset: str, **kwargs) -> GenerationResult:
    if _prefer_mocks():
        from tests.mocks import run_generation as _rg
    else:
        from hx.generation import run_generation as _rg
    return _rg(harness_id, dataset, **kwargs)


def _load_harness(harness_id: str):
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


def _noise(dataset: str) -> float:
    if _prefer_mocks():
        from tests.mocks import get_noise_floor
        return float(get_noise_floor(dataset))
    try:
        from hx.scorer import get_noise_floor
        return float(get_noise_floor(dataset))
    except Exception:
        return DEFAULT_NOISE


def _persist_reports(reports: list[Report]) -> None:
    try:
        from hx.config import get_client

        get_client("admin")["hx_evidence"]["reports"].insert_many(
            [r.model_dump() for r in reports]
        )
    except Exception:
        try:
            from tests.mocks import STORE

            STORE.reports.extend([r.model_dump() for r in reports])
        except Exception:
            pass


def _overview_with_splits(harness_id: str, base: GenerationResult) -> dict[str, Any]:
    from hx.overview import build_overview

    overview = build_overview(harness_id)
    failed_tasks = [tid for tid, s in base.train_outcomes.items() if s < 1.0]
    passed_tasks = [tid for tid, s in base.train_outcomes.items() if s >= 1.0]
    overview["failed_train_task_ids"] = failed_tasks
    overview["passed_train_task_ids"] = passed_tasks
    overview["passed_train_trace_ids"] = list(base.passed_train_trace_ids)
    overview["failed_train_trace_ids"] = list(base.failed_train_trace_ids)
    overview.pop("heldout_outcomes", None)
    return overview


def _fill_actuals(parent: GenerationResult, child: GenerationResult, manifest: Manifest) -> Manifest:
    p, c = parent.train_outcomes, child.train_outcomes
    ids = set(p) | set(c)
    actual_fixed = [t for t in ids if p.get(t, 0) < 1.0 and c.get(t, 0) >= 1.0]
    actual_regressed = [t for t in ids if p.get(t, 1) >= 1.0 and c.get(t, 1) < 1.0]
    return manifest.model_copy(update={"actual_fixed": actual_fixed, "actual_regressed": actual_regressed})


def evolve(dataset: str, start_harness_id: str, generations: int = 10) -> str:
    hid = start_harness_id
    start, _ = _load_harness(hid)
    start_gen = start.generation
    rng = random.Random(0)
    for i in range(1, generations + 1):
        gen = start_gen + i
        phase = _cadence(gen)
        base = _run_generation(hid, dataset)
        from hx.debugger import analyze

        pass_sample = list(base.passed_train_trace_ids)
        rng.shuffle(pass_sample)
        trace_ids = list(base.failed_train_trace_ids) + pass_sample[:5]
        reports = analyze(hid, trace_ids)
        _persist_reports(reports)
        if phase == "prune":
            from hx.ablate import prune_candidates

            prune_candidates(hid, dataset)
        overview = _overview_with_splits(hid, base)
        overview["noise"] = _noise(dataset)
        from hx.evolve import propose_edits

        manifests = propose_edits(hid, phase, overview)
        results: list[GenerationResult] = []
        evaluated: list[Manifest] = []
        for m in manifests:
            child_h = _apply_edits(hid, [m.edit], phase, gen)
            m = m.model_copy(update={"child_harness_id": child_h.harness_id})
            child_r = _run_generation(child_h.harness_id, dataset)
            noise = _noise(dataset)
            ok, reason = gate(base, child_r, m.edit.op, noise)
            m = _fill_actuals(base, child_r, m)
            m = m.model_copy(
                update={
                    "gate_reason": reason,
                    "verdict": "promoted" if ok else "rejected",
                }
            )
            evaluated.append(m)
            results.append(child_r)
        winners = [m for m in evaluated if m.verdict == "promoted"]
        try:
            from tests.mocks import STORE

            STORE.manifests.extend(evaluated)
        except Exception:
            pass
        set_manifests(evaluated)
        hid = merge_promoted(hid, winners, gen, phase).harness_id
        merged, comps = _load_harness(hid)
        n_rules = sum(1 for c in comps if c.kind == "rule")
        scores = prediction_scores(gen)
        log.info(
            "gen=%s phase=%s train=%.3f heldout=%.3f tokens=%.1f n_rules=%s promoted=%s rejected=%s pred=%s",
            gen,
            phase,
            merged.train_score or 0.0,
            merged.heldout_score or 0.0,
            merged.tokens_per_task or 0.0,
            n_rules,
            len(winners),
            len(evaluated) - len(winners),
            {k: scores[k] for k in ("fix_precision", "fix_recall") if k in scores},
        )
        print(
            f"gen={gen} phase={phase} train={merged.train_score} heldout={merged.heldout_score} "
            f"tokens={merged.tokens_per_task} n_rules={n_rules} "
            f"promoted={len(winners)} rejected={len(evaluated) - len(winners)}"
        )
    return hid
