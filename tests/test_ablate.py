from __future__ import annotations

from hx.ablate import prune_candidates
from tests.mocks import STORE, load_harness, reset_store


def setup_function() -> None:
    reset_store()


def test_pinned_never_ablated() -> None:
    abs_ = prune_candidates("h0", "sample_mflix")
    ids = {a.component_id for a in abs_}
    assert "rule.base" not in ids


def test_unused_tool_skipped_without_task_ids() -> None:
    abs_ = prune_candidates("h0", "sample_mflix")
    unused = [a for a in abs_ if a.component_id == "tool.unused"]
    assert len(unused) == 1
    assert unused[0].n_tasks == 0
    assert unused[0].delta == 0.0


def test_rule_ablation_uses_context_component_tasks() -> None:
    abs_ = prune_candidates("h0", "sample_mflix")
    types = [a for a in abs_ if a.component_id == "rule.types"]
    assert types
    assert types[0].n_tasks >= 1
    parent, comps = load_harness("h0")
    assert any(c.component_id == "rule.types" for c in comps)
    assert STORE.ablations
