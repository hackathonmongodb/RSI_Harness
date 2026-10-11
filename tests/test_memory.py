from hx.memory import cases as case_ops
from hx.memory import harness as rule_ops
from hx.memory.models import RULE_CAP, TurnTrace, ToolEvent
from hx.memory.recall import select_cases
from hx.memory.reflect import Review, apply_review, redact, should_review
from hx.memory.store import LocalStore
from tests.conftest import FakeLLM


def test_store_starts_with_empty_v0(store):
    hv = store.current_harness()
    assert hv.version == 0 and hv.rules == []
    assert (store.home / "harness" / "v0000.json").exists()


def test_rule_lifecycle_trial_to_active(store):
    msg = rule_ops.add_rule(store, "No interactive runs", "Never run TUI/GUI programs directly.", "hung")
    assert msg.startswith("rule+ no-interactive-runs (trial)")
    hv = store.current_harness()
    assert hv.version == 1 and hv.parent == 0
    assert hv.get("no-interactive-runs").status == "trial"
    out = rule_ops.judge_rules(store, {"no-interactive-runs": "helped"})
    assert out and "promote" in out[0]
    hv = store.current_harness()
    assert hv.version == 2 and hv.get("no-interactive-runs").status == "active"
    assert "no-interactive-runs" in (store.home / "HARNESS.md").read_text()


def test_trial_rule_retired_after_two_strikes(store):
    rule_ops.add_rule(store, "r", "Do X.")
    assert rule_ops.judge_rules(store, {"r": "violated"}) == []   # counter only, no new version
    assert store.current_harness().version == 1
    assert store.current_harness().get("r").strikes == 1
    out = rule_ops.judge_rules(store, {"r": "violated"})
    assert "retire rule r" in out[0]
    assert store.current_harness().live_rules() == []


def test_irrelevant_verdict_changes_nothing(store):
    rule_ops.add_rule(store, "r", "Do X.")
    rule_ops.judge_rules(store, {"r": "irrelevant"})
    r = store.current_harness().get("r")
    assert r.trials == 0 and r.status == "trial"


def test_rule_cap(store):
    for i in range(RULE_CAP):
        rule_ops.add_rule(store, f"r{i}", f"rule {i}")
    assert "harness is full" in rule_ops.add_rule(store, "extra", "one more")


def test_add_existing_rule_id_revises(store):
    rule_ops.add_rule(store, "r", "Do X.")
    msg = rule_ops.add_rule(store, "r", "Do X carefully.")
    assert msg.startswith("rule~ r")
    assert store.current_harness().get("r").text == "Do X carefully."


def test_case_roundtrip_and_index(store):
    msg = case_ops.file_case(store, "Curses without TTY", "curses programs crash with setupterm error when run by the agent's shell",
                             "**Symptom:** _curses.error\n**How to apply:** test logic headlessly", ["curses"])
    assert msg == "case+ curses-without-tty"
    c = store.get_case("curses-without-tty")
    assert c.tags == ["curses"] and "Symptom" in c.body
    # duplicate title gets a unique id
    assert case_ops.file_case(store, "Curses without TTY", "d", "b") == "case+ curses-without-tty-2"
    assert "curses-without-tty:" in (store.home / "CASES.md").read_text()
    case_ops.update_case(store, "curses-without-tty-2", retire=True)
    assert [c.case_id for c in store.list_cases()] == ["curses-without-tty"]


def test_case_retired_when_misleading_twice(store):
    case_ops.file_case(store, "t", "d", "b", case_id="x")
    case_ops.judge_cases(store, {"x": "misleading"})
    assert store.get_case("x").status == "active"
    case_ops.judge_cases(store, {"x": "misleading"})
    assert store.get_case("x").status == "retired"


def test_recall_filters_invalid_ids_and_skips_empty(store):
    llm = FakeLLM(json_handler=lambda s, u: {"selected": ["a", "ghost", "a"]})
    assert select_cases(llm, "anything", []) == []
    assert llm.json_calls == []
    case_ops.file_case(store, "A", "about pygame", "b", case_id="a")
    case_ops.file_case(store, "B", "about docker", "b", case_id="b")
    assert select_cases(llm, "make a pygame game", store.list_cases()) == ["a"]
    assert select_cases(llm, "x", store.list_cases(), exclude={"a"}) == []


def test_recall_survives_llm_error(store):
    def boom(s, u):
        raise RuntimeError("down")

    case_ops.file_case(store, "A", "d", "b", case_id="a")
    assert select_cases(FakeLLM(json_handler=boom), "q", store.list_cases()) == []


def _trace(**kw) -> TurnTrace:
    base = dict(task="build snake", steps=2)
    base.update(kw)
    return TurnTrace(**base)


def test_should_review():
    from hx.memory.models import HarnessVersion, Rule

    hv = HarnessVersion(version=1, rules=[Rule(rule_id="t", text="x", status="trial")])
    clean = _trace(events=[ToolEvent(step=1, tool="bash", args="ls", ok=True)])
    assert not should_review(clean, HarnessVersion(version=0))
    assert should_review(clean.model_copy(update={"rule_ids": ["t"]}), hv)
    failed = _trace(events=[ToolEvent(step=1, tool="bash", args="python x.py", ok=False, kind="nonzero_exit", benign=False)])
    assert should_review(failed, HarnessVersion(version=0))
    benign = _trace(events=[ToolEvent(step=1, tool="bash", args="grep x", ok=False, kind="nonzero_exit", benign=True)])
    assert not should_review(benign, HarnessVersion(version=0))
    assert should_review(_trace(user_feedback="snake walks through walls"), HarnessVersion(version=0))


def test_apply_review_writes_one_item_and_gates(store):
    rule_ops.add_rule(store, "old", "Old rule.")
    trace = _trace(rule_ids=["old"])
    review = Review.from_json({
        "analysis": "ran curses app without TTY",
        "action": "file_case",
        "case": {"id": "curses-tty", "title": "curses needs a TTY", "description": "curses apps fail in agent shell",
                 "body": "**Symptom:** setupterm\n**Root cause:** no tty", "tags": ["curses"]},
        "rule_verdicts": {"old": "helped", "not-loaded": "violated"},
    })
    actions = apply_review(store, review, trace)
    assert "case+ curses-tty" in actions
    assert any("promote rule old" in a for a in actions)
    assert store.get_case("curses-tty") is not None


def test_review_parsing_defaults_bad_action():
    r = Review.from_json({"action": "delete_everything", "rule_verdicts": "nope"})
    assert r.action == "none" and r.rule_verdicts == {}


def test_redact_secrets():
    s = redact("key gsk_" + "a" * 40 + " uri mongodb+srv://u:p@host/x")
    assert "gsk_" not in s and "u:p@" not in s


def test_local_store_survives_corrupt_files(tmp_path):
    st = LocalStore(tmp_path / "h")
    (st.home / "cases" / "bad.md").write_text("no frontmatter")
    (st.home / "harness" / "v0009.json").write_text("{broken")
    assert st.list_cases() == []
    assert st.current_harness().version == 0
