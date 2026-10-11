from __future__ import annotations

import inspect

from hx.models import Edit, Task
from tests import mocks
from tests.mocks import (
    apply_edits,
    complete,
    get_noise_floor,
    load_harness,
    reset_store,
    run_generation,
    score,
)


def setup_function() -> None:
    reset_store()


def test_signatures_match_spec_section_7() -> None:
    assert inspect.signature(score) == inspect.signature(mocks.score)
    sig_score = inspect.signature(score)
    assert list(sig_score.parameters) == ["task", "parsed"]
    assert list(inspect.signature(get_noise_floor).parameters) == ["dataset"]
    h = inspect.signature(load_harness)
    assert list(h.parameters) == ["harness_id"]
    a = inspect.signature(apply_edits)
    assert list(a.parameters) == ["parent_id", "edits", "phase", "generation"]
    g = inspect.signature(run_generation)
    assert list(g.parameters) == ["harness_id", "dataset", "seeds", "parallel", "task_ids"]
    assert g.parameters["seeds"].default == [0]
    assert g.parameters["parallel"].default == 8
    assert g.parameters["task_ids"].default is None
    c = inspect.signature(complete)
    assert list(c.parameters) == ["model", "system", "messages", "tools", "temperature", "max_tokens"]
    assert c.parameters["tools"].default is None
    assert c.parameters["temperature"].default == 0.0
    assert c.parameters["max_tokens"].default == 2048


def test_get_noise_floor_is_default_margin() -> None:
    assert get_noise_floor("sample_mflix") == 0.05


def test_run_generation_respects_task_ids_and_train_outcomes() -> None:
    result = run_generation("h0", "sample_mflix", task_ids=["train_fail_0", "train_pass_0"])
    assert set(result.train_outcomes) == {"train_fail_0", "train_pass_0"}
    assert result.train_outcomes["train_fail_0"] == 0.0
    assert result.train_outcomes["train_pass_0"] == 1.0
    assert "held_0" not in result.train_outcomes
    assert result.heldout_outcomes == {}


def test_apply_edits_returns_child_harness() -> None:
    child = apply_edits(
        "h0",
        [Edit(op="delete", kind="rule", target_component_id="rule.types")],
        "prune",
        1,
    )
    assert child.parent_id == "h0"
    assert child.status == "candidate"
    ids = {r.component_id for r in child.components}
    assert "rule.types" not in ids
    parent, _ = load_harness("h0")
    assert any(r.component_id == "rule.types" for r in parent.components)


def test_score_parse_none() -> None:
    task = Task(
        task_id="train_fail_0",
        dataset="sample_mflix",
        split="train",
        question="q",
        collection="movies",
    )
    s, err, _ = score(task, None)
    assert s == 0.0
    assert err == "parse"
