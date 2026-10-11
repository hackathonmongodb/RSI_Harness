from __future__ import annotations

import json

from hx.loop import _cadence, evolve
from hx.models import Edit
from hx.promote import merge_promoted
from tests.mocks import STORE, reset_store


def setup_function() -> None:
    reset_store()


def test_cadence_order() -> None:
    assert _cadence(1) == "policy"
    assert _cadence(4) == "prune"


def test_merge_promoted_calls_apply_edits_with_all_winners() -> None:
    from hx.models import Component, Manifest

    c = Component(component_id="policy.cot", kind="policy", name="cot", value=True)
    m = Manifest(
        edit_id="e1",
        generation=1,
        phase="policy",
        parent_harness_id="h0",
        edit=Edit(op="modify", kind="policy", target_component_id="policy.cot", component=c),
        root_cause="parse",
        targeted_fix="cot",
        evidence_trace_ids=["tr_train_fail_0"],
        predicted_fixes=["train_fail_0"],
        verdict="promoted",
    )
    child = merge_promoted("h0", [m], 1, "policy")
    assert child.parent_id == "h0"
    assert child.status == "promoted"


def test_evolve_two_generations_on_mocks() -> None:
    STORE.llm_oracle["overview"] = json.dumps(
        {
            "edits": [
                {
                    "op": "modify",
                    "kind": "policy",
                    "target_component_id": "policy.cot",
                    "value": True,
                    "root_cause": "parse errors",
                    "targeted_fix": "enable cot",
                    "evidence_trace_ids": ["tr_train_fail_0"],
                    "predicted_fixes": ["train_fail_0"],
                    "predicted_regressions": ["train_pass_0"],
                }
            ]
        }
    )
    hid = evolve("sample_mflix", "h0", generations=2)
    assert hid
    assert hid in STORE.harnesses or hid == "h0"
    for payload in STORE.last_complete_payloads:
        blob = payload["system"] + json.dumps(payload["messages"])
        assert "heldout_outcomes" not in blob
        assert "HELD OUT" not in blob
        assert "hx_eval.gold" not in blob


def test_reports_written_by_loop_store_not_debugger_client() -> None:
    import hx.debugger as d
    import inspect

    assert "get_client" not in inspect.getsource(d)
    evolve("sample_mflix", "h0", generations=1)
    assert STORE.reports  # loop persist path
