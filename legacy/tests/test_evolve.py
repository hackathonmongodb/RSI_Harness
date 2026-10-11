from __future__ import annotations

from hx.evolve import (
    PINNED_IDS,
    SOFT_RULE_CAP,
    propose_edits,
    validate_edit,
    validate_manifest,
)
from hx.loop import _cadence
from hx.models import Component, Edit, Manifest
from tests.mocks import STORE, reset_store


def setup_function() -> None:
    reset_store()


def _comps() -> list[Component]:
    _, comps = __import__("tests.mocks", fromlist=["load_harness"]).load_harness("h0")
    return comps


def test_cadence_policy_growth_growth_prune() -> None:
    assert [_cadence(i) for i in range(1, 9)] == [
        "policy",
        "growth",
        "growth",
        "prune",
        "policy",
        "growth",
        "growth",
        "prune",
    ]
    assert _cadence(0) == "baseline"


def test_reject_pinned() -> None:
    edit = Edit(op="modify", kind="rule", target_component_id="rule.base")
    assert validate_edit(edit, "growth", _comps(), 2)
    assert "rule.base" in PINNED_IDS


def test_reject_out_of_range_policy() -> None:
    c = Component(component_id="policy.cot", kind="policy", name="cot", value=7)
    edit = Edit(op="modify", kind="policy", target_component_id="policy.cot", component=c)
    assert validate_edit(edit, "policy", _comps(), 2)
    c2 = Component(component_id="policy.doc_truncate_chars", kind="policy", name="t", value=50)
    edit2 = Edit(op="modify", kind="policy", target_component_id="policy.doc_truncate_chars", component=c2)
    assert validate_edit(edit2, "policy", _comps(), 2)


def test_accept_in_range_policy() -> None:
    c = Component(component_id="policy.cot", kind="policy", name="cot", value=True)
    edit = Edit(op="modify", kind="policy", target_component_id="policy.cot", component=c)
    assert validate_edit(edit, "policy", _comps(), 2) is None


def test_rule_content_cap() -> None:
    c = Component(component_id="rule.x", kind="rule", name="x", content="a" * 401)
    edit = Edit(op="add", kind="rule", target_component_id="rule.x", component=c)
    assert validate_edit(edit, "growth", _comps(), 2)


def test_soft_rule_cap() -> None:
    c = Component(component_id="rule.x", kind="rule", name="x", content="ok")
    edit = Edit(op="add", kind="rule", target_component_id="rule.x", component=c)
    assert validate_edit(edit, "growth", _comps(), SOFT_RULE_CAP)


def test_no_evidence_rejected() -> None:
    c = Component(component_id="policy.cot", kind="policy", name="cot", value=True)
    edit = Edit(op="modify", kind="policy", target_component_id="policy.cot", component=c)
    m = Manifest(
        edit_id="e",
        generation=1,
        phase="policy",
        parent_harness_id="h0",
        edit=edit,
        evidence_trace_ids=[],
        root_cause="parse",
        targeted_fix="cot on",
        predicted_fixes=["train_fail_0"],
    )
    err = validate_manifest(m, "policy", _comps(), {"train_fail_0"}, {"train_pass_0"}, 2)
    assert err and "evidence" in err


def test_invented_prediction_ids_rejected() -> None:
    c = Component(component_id="policy.cot", kind="policy", name="cot", value=True)
    edit = Edit(op="modify", kind="policy", target_component_id="policy.cot", component=c)
    m = Manifest(
        edit_id="e",
        generation=1,
        phase="policy",
        parent_harness_id="h0",
        edit=edit,
        evidence_trace_ids=["tr_train_fail_0"],
        root_cause="parse",
        targeted_fix="cot on",
        predicted_fixes=["not_a_real_task"],
        predicted_regressions=["also_fake"],
    )
    err = validate_manifest(m, "policy", _comps(), {"train_fail_0"}, {"train_pass_0"}, 2)
    assert err and "predicted_fixes" in err


def test_phase_cap_truncates_propose_edits() -> None:
    STORE.llm_oracle["overview"] = __import__("json").dumps(
        {
            "edits": [
                {
                    "op": "modify",
                    "kind": "policy",
                    "target_component_id": "policy.cot",
                    "value": True,
                    "root_cause": "parse",
                    "targeted_fix": "cot",
                    "evidence_trace_ids": ["tr_train_fail_0"],
                    "predicted_fixes": ["train_fail_0"],
                    "predicted_regressions": ["train_pass_0"],
                },
                {
                    "op": "modify",
                    "kind": "policy",
                    "target_component_id": "policy.max_retries",
                    "value": 1,
                    "root_cause": "exec",
                    "targeted_fix": "retry",
                    "evidence_trace_ids": ["tr_train_fail_1"],
                    "predicted_fixes": ["train_fail_1"],
                    "predicted_regressions": [],
                },
                {
                    "op": "modify",
                    "kind": "policy",
                    "target_component_id": "policy.include_indexes",
                    "value": True,
                    "root_cause": "scan",
                    "targeted_fix": "indexes",
                    "evidence_trace_ids": ["tr_train_fail_2"],
                    "predicted_fixes": ["train_fail_2"],
                    "predicted_regressions": [],
                },
            ]
        }
    )
    overview = {
        "failed_train_task_ids": ["train_fail_0", "train_fail_1", "train_fail_2"],
        "passed_train_task_ids": ["train_pass_0", "train_pass_1"],
        "passed_train_trace_ids": ["tr_train_pass_0"],
    }
    ms = propose_edits("h0", "policy", overview)
    assert len(ms) <= 2
