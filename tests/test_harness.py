from hx.harness import H0, H_MONGODB, apply_edits, assemble
from hx.models import Edit


def test_h0_and_h_mongodb_prompts_differ():
    q = "Show me the genres and runtime of 10 movies from 2015 that have the most comments"
    a = assemble(H0, q)
    b = assemble(H_MONGODB, q)
    assert a.system != b.system
    assert "schema" not in a.context_components
    assert "samples" not in a.context_components
    assert "indexes" not in a.context_components
    assert "schema" in b.context_components
    assert "samples" in b.context_components
    assert "indexes" in b.context_components
    assert "Annotated schema" in b.system
    assert "Annotated schema" not in a.system
    assert "Sample documents" in b.system
    assert "Sample documents" not in a.system


def test_delete_rule_removes_exact_text_only():
    h = H0.model_copy(deep=True)
    h.rules = ["keep me", "delete me", "also keep"]
    policy_before = h.policy.model_dump()
    components_before = [c.model_dump() for c in h.components]
    out = apply_edits(h, [Edit(op="delete_rule", value="delete me")])
    assert out.rules == ["keep me", "also keep"]
    assert "delete me" not in out.rules
    assert out.policy.model_dump() == policy_before
    assert [c.model_dump() for c in out.components] == components_before
    assert out.version == h.version + 1
    assert out.parent_version == h.version


def test_set_policy_changes_only_that_knob():
    h = H_MONGODB.model_copy(deep=True)
    before = h.policy.model_dump()
    rules_before = list(h.rules)
    out = apply_edits(h, [Edit(op="set_policy", knob="max_retries", value=9)])
    assert out.policy.max_retries == 9
    for k, v in before.items():
        if k == "max_retries":
            continue
        assert getattr(out.policy, k) == v
    assert out.rules == rules_before
