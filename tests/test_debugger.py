from __future__ import annotations

import inspect
import json

from hx.debugger import analyze
from hx.models import Task
from tests.fixtures.attribution_cases import CASES, case_to_trace
from tests.mocks import STORE, reset_store


def setup_function() -> None:
    reset_store()


def test_analyze_does_not_call_get_client() -> None:
    import hx.debugger as d

    src = inspect.getsource(d)
    assert "get_client" not in src


def test_invalid_json_retried_then_dropped() -> None:
    STORE.llm_oracle.clear()
    STORE.llm_oracle["raw_output"] = "not-json"
    # complete returns this whenever needle matches; second call still not-json
    reports = analyze("h0", ["tr_train_fail_0"])
    assert reports == []


def test_valid_json_becomes_report() -> None:
    STORE.llm_oracle["raw_output"] = json.dumps(
        {
            "root_cause": "types",
            "implicated_component_id": "rule.types",
            "implicated_kind": "rule",
            "missing_component": False,
            "confidence": 0.9,
            "evidence": "Decimal128",
        }
    )
    reports = analyze("h0", ["tr_train_fail_0"])
    assert len(reports) == 1
    assert reports[0].implicated_component_id == "rule.types"
    assert reports[0].missing_component is False


def test_missing_component_clears_implicated_id() -> None:
    STORE.llm_oracle["raw_output"] = json.dumps(
        {
            "root_cause": "need lookup",
            "implicated_component_id": "rule.types",
            "implicated_kind": "rule",
            "missing_component": True,
            "confidence": 0.7,
            "evidence": "no $lookup",
        }
    )
    reports = analyze("h0", ["tr_train_fail_1"])
    assert reports[0].missing_component is True
    assert reports[0].implicated_component_id is None


def test_attribution_suite_oracle_wiring() -> None:
    """Fake complete oracles by case id so parse/validate wiring is proven on 10 labels."""
    for case in CASES:
        tid = f"attr_{case['id']}"
        STORE.tasks[tid] = Task(
            task_id=tid,
            dataset="sample_mflix",
            split="train",
            question=case["question"],
            collection="movies",
        )
        tr = case_to_trace(case)
        STORE.traces[tr.trace_id] = tr
        expected = {
            "root_cause": case["error"],
            "implicated_component_id": case["implicated_component_id"],
            "implicated_kind": "policy" if (case["implicated_component_id"] or "").startswith("policy.") else "rule",
            "missing_component": case["missing_component"],
            "confidence": 0.85,
            "evidence": case["error"],
        }
        STORE.llm_oracle[case["question"]] = json.dumps(expected)

    for case in CASES:
        tr = case_to_trace(case)
        reports = analyze("h0", [tr.trace_id])
        assert len(reports) == 1, case["id"]
        r = reports[0]
        if case["missing_component"]:
            assert r.missing_component, case["id"]
            assert r.implicated_component_id is None, case["id"]
        else:
            assert r.implicated_component_id == case["implicated_component_id"], case["id"]


def test_prompts_exclude_gold_and_heldout() -> None:
    analyze("h0", ["tr_train_fail_0"])
    for payload in STORE.last_complete_payloads:
        blob = payload["system"] + json.dumps(payload["messages"])
        assert "heldout_outcomes" not in blob
        assert "HELD OUT" not in blob
        assert "hx_eval.gold" not in blob
        assert "Gold.pipeline" not in blob
