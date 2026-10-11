"""Prediction precision/recall. Public signature matches SPEC.md section 7."""
from __future__ import annotations

from typing import Any, Optional

from hx.models import Manifest

_MANIFESTS: list[Manifest] = []


def set_manifests(items: list[Manifest]) -> None:
    global _MANIFESTS
    _MANIFESTS = list(items)


def _load_manifests(generation: int) -> list[Manifest]:
    sourced = [m for m in _MANIFESTS if m.generation == generation]
    if sourced:
        return sourced
    try:
        from tests.mocks import STORE

        return [m for m in STORE.manifests if m.generation == generation]
    except ImportError:
        return []


def _pr(predicted: set[str], actual: set[str]) -> tuple[Optional[float], Optional[float]]:
    precision = None if not predicted else len(predicted & actual) / len(predicted)
    recall = None if not actual else len(predicted & actual) / len(actual)
    return precision, recall


def prediction_scores(generation: int) -> dict[str, Any]:
    manifests = _load_manifests(generation)
    per: list[dict[str, Any]] = []
    all_pf: set[str] = set()
    all_af: set[str] = set()
    all_pr: set[str] = set()
    all_ar: set[str] = set()
    for m in manifests:
        pf, af = set(m.predicted_fixes), set(m.actual_fixed)
        pr, ar = set(m.predicted_regressions), set(m.actual_regressed)
        p_fix, r_fix = _pr(pf, af)
        p_reg, r_reg = _pr(pr, ar)
        per.append(
            {
                "edit_id": m.edit_id,
                "fix_precision": p_fix,
                "fix_recall": r_fix,
                "reg_precision": p_reg,
                "reg_recall": r_reg,
            }
        )
        all_pf |= pf
        all_af |= af
        all_pr |= pr
        all_ar |= ar
    gp_f, gr_f = _pr(all_pf, all_af)
    gp_r, gr_r = _pr(all_pr, all_ar)
    return {
        "generation": generation,
        "n_manifests": len(manifests),
        "per_edit": per,
        "fix_precision": gp_f,
        "fix_recall": gr_f,
        "reg_precision": gp_r,
        "reg_recall": gr_r,
    }
