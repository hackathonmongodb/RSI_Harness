"""Promotion gate and merge. Public signatures match SPEC.md section 7."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from hx.models import GenerationResult, Harness, Manifest, Op, Phase

DEFAULT_NOISE = 0.05

if TYPE_CHECKING:
    pass


def gate(
    parent: GenerationResult,
    child: GenerationResult,
    op: Op,
    noise: float,
) -> tuple[bool, str]:
    """Return (pass, reason). `noise` is the promotion margin (default 0.05 until A lands)."""
    margin = DEFAULT_NOISE if noise is None else noise
    p_train, c_train = parent.train_score, child.train_score
    p_held, c_held = parent.heldout_score, child.heldout_score
    p_tok, c_tok = parent.tokens_per_task, child.tokens_per_task

    if op in ("add", "modify"):
        held_ok = c_held >= p_held - margin
        train_ok = c_train >= p_train + margin
        ok = held_ok and train_ok
        reason = (
            f"op={op} heldout {c_held:.4f} vs parent {p_held:.4f} (need >= {p_held - margin:.4f}): "
            f"{'ok' if held_ok else 'fail'}; "
            f"train {c_train:.4f} vs parent {p_train:.4f} (need >= {p_train + margin:.4f}): "
            f"{'ok' if train_ok else 'fail'}"
        )
        return ok, reason

    if op == "delete":
        held_ok = c_held >= p_held - margin
        train_ok = c_train >= p_train - margin
        tok_ok = c_tok < p_tok
        ok = held_ok and train_ok and tok_ok
        reason = (
            f"op=delete heldout {c_held:.4f} vs {p_held:.4f} (need >= {p_held - margin:.4f}): "
            f"{'ok' if held_ok else 'fail'}; "
            f"train {c_train:.4f} vs {p_train:.4f} (need >= {p_train - margin:.4f}): "
            f"{'ok' if train_ok else 'fail'}; "
            f"tokens {c_tok:.4f} < {p_tok:.4f}: {'ok' if tok_ok else 'fail'}"
        )
        return ok, reason

    return False, f"unknown op {op}"


def _prefer_mocks() -> bool:
    return os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}


def _apply_edits(parent_id: str, edits, phase: Phase, generation: int) -> Harness:
    if _prefer_mocks():
        from tests.mocks import apply_edits as _real
    else:
        from hx.harness import apply_edits as _real
    return _real(parent_id, edits, phase, generation)


def _load_harness(harness_id: str):
    if _prefer_mocks():
        from tests.mocks import load_harness as _real
    else:
        from hx.harness import load_harness as _real
    return _real(harness_id)


def merge_promoted(
    parent_id: str,
    winners: list[Manifest],
    generation: int,
    phase: Phase,
) -> Harness:
    """Apply all winning edits in one child; mark other candidates rejected (in-memory until Atlas)."""
    edits = [m.edit for m in winners]
    if not edits:
        parent, _ = _load_harness(parent_id)
        return parent
    child = _apply_edits(parent_id, edits, phase, generation)
    try:
        from tests.mocks import STORE

        STORE.harnesses[child.harness_id] = child.model_copy(update={"status": "promoted"})
        child = STORE.harnesses[child.harness_id]
        winner_children = {m.child_harness_id for m in winners if m.child_harness_id}
        for hid, h in list(STORE.harnesses.items()):
            if h.parent_id == parent_id and h.status == "candidate" and hid not in winner_children and hid != child.harness_id:
                STORE.harnesses[hid] = h.model_copy(update={"status": "rejected"})
    except Exception:
        pass
    return child.model_copy(update={"status": "promoted"})
