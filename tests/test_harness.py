from hx.harness import STUB_CTX, apply_edits, assemble, load_harness
from hx.models import Component, Edit, Task

TASK = Task(
    task_id="q",
    dataset="sample_mflix",
    split="train",
    question="Show the genres and runtime of 10 movies from 2015 that have the most comments. Return genres, runtime.",
    collection="movies",
)


def test_h0_and_h_mongodb_prompts_differ():
    h0, c0 = load_harness("h0")
    hm, cm = load_harness("h_mongodb")
    s0, m0, _ = assemble(h0, c0, TASK, STUB_CTX)
    s1, m1, _ = assemble(hm, cm, TASK, STUB_CTX)
    assert s0 != s1
    u0, u1 = m0[0]["content"], m1[0]["content"]
    assert "Example documents" not in u0
    assert "Example documents" in u1
    assert "Indexes:" not in u0
    assert "Indexes:" in u1
    assert "Think step by step" not in s0
    assert "Think step by step" in s1


def test_delete_rule_removes_exact_text_only():
    parent, comps = load_harness("h_mongodb")
    target = next(c for c in comps if c.component_id == "rule.tip.dates")
    policies_before = {c.component_id: c.value for c in comps if c.kind == "policy"}
    child = apply_edits(
        parent.harness_id,
        [Edit(op="delete", kind="rule", target_component_id="rule.tip.dates")],
        "prune",
        1,
    )
    _, after = load_harness(child.harness_id)
    sys, _, _ = assemble(child, after, TASK, STUB_CTX)
    assert target.content not in sys
    assert "aggregation pipelines" in sys  # rule.base remains
    policies_after = {c.component_id: c.value for c in after if c.kind == "policy"}
    assert policies_after == policies_before
    ids = {c.component_id for c in after}
    assert "rule.tip.dates" not in ids


def test_set_policy_changes_only_that_knob():
    parent, comps = load_harness("h0")
    before = {c.component_id: c.value for c in comps if c.kind == "policy"}
    rules_before = {c.component_id: c.content for c in comps if c.kind == "rule"}
    child = apply_edits(
        parent.harness_id,
        [
            Edit(
                op="modify",
                kind="policy",
                target_component_id="policy.max_retries",
                component=Component(
                    component_id="policy.max_retries",
                    kind="policy",
                    name="max_retries",
                    value=2,
                ),
            )
        ],
        "policy",
        1,
    )
    _, after = load_harness(child.harness_id)
    after_p = {c.component_id: c.value for c in after if c.kind == "policy"}
    assert after_p["policy.max_retries"] == 2
    for k, v in before.items():
        if k != "policy.max_retries":
            assert after_p[k] == v
    rules_after = {c.component_id: c.content for c in after if c.kind == "rule"}
    assert rules_after == rules_before
