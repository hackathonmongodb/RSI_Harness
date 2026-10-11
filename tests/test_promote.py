from __future__ import annotations

from hx.models import GenerationResult
from hx.promote import DEFAULT_NOISE, gate


def _gr(train: float, held: float, tokens: float = 100.0) -> GenerationResult:
    return GenerationResult(
        harness_id="h",
        train_score=train,
        heldout_score=held,
        tokens_per_task=tokens,
        train_outcomes={},
        heldout_outcomes={},
        failed_train_trace_ids=[],
        passed_train_trace_ids=[],
    )


def test_default_noise() -> None:
    assert DEFAULT_NOISE == 0.05


def test_gate_add_needs_real_train_gain_and_heldout_within_noise() -> None:
    parent = _gr(0.40, 0.50)
    child_ok = _gr(0.46, 0.48)
    ok, reason = gate(parent, child_ok, "add", 0.05)
    assert ok, reason
    child_wobble = _gr(0.41, 0.50)
    ok, _ = gate(parent, child_wobble, "modify", 0.05)
    assert not ok
    child_held_drop = _gr(0.60, 0.40)
    ok, _ = gate(parent, child_held_drop, "add", 0.05)
    assert not ok


def test_gate_delete_allows_train_wobble_requires_fewer_tokens() -> None:
    parent = _gr(0.50, 0.50, tokens=100.0)
    child_ok = _gr(0.48, 0.49, tokens=80.0)
    ok, reason = gate(parent, child_ok, "delete", 0.05)
    assert ok, reason
    child_tokens_same = _gr(0.50, 0.50, tokens=100.0)
    ok, _ = gate(parent, child_tokens_same, "delete", 0.05)
    assert not ok
    child_held_crash = _gr(0.50, 0.40, tokens=10.0)
    ok, _ = gate(parent, child_held_crash, "delete", 0.05)
    assert not ok
