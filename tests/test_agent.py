"""End-to-end inner loop with a scripted model: fail -> self-review -> memory -> retry."""
import json

from hx.agent import CLEARED, Agent, compact, context_tokens
from hx.llm import Reply
from hx.memory import cases as case_ops
from hx.memory.reflect import REVIEW_SYSTEM
from hx.memory.recall import SELECT_SYSTEM
from tests.conftest import FakeLLM, call


def review_handler(review: dict, selected: list[str] | None = None):
    def handler(system, user):
        if system == REVIEW_SYSTEM:
            return review
        if system == SELECT_SYSTEM:
            return {"selected": selected or []}
        return {}
    return handler


def test_clean_turn_writes_trace_and_skips_review(settings, store):
    llm = FakeLLM([
        Reply(tool_calls=[call("write_file", path="hello.py", content="print('hi')\n")]),
        Reply(tool_calls=[call("bash", command="python3 hello.py")]),
        Reply(content="Created hello.py; run `python3 hello.py`."),
    ])
    agent = Agent(settings=settings, llm=llm, store=store)
    trace = agent.run_turn("write hello world")
    assert trace.outcome == "ok" and trace.steps == 2
    assert (settings.workdir / "hello.py").read_text() == "print('hi')\n"
    assert llm.json_calls == []                       # nothing failed, nothing to judge
    assert store.get_trace(trace.trace_id).final_text.startswith("Created")
    # tool messages are paired with their calls
    msgs = llm.chat_calls[-1]
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "tool", "assistant", "tool"]


def test_failure_triggers_review_and_files_case(settings, store):
    review = {
        "analysis": "Ran a curses program in a non-TTY shell.",
        "action": "file_case",
        "case": {"id": "curses-no-tty", "title": "curses needs a TTY",
                 "description": "curses apps crash with _curses.error when run through the agent's bash tool",
                 "body": "**Symptom:** _curses.error\n**Root cause:** no TTY\n**When it applies:** curses\n"
                         "**How to apply:** verify with python -m py_compile instead", "tags": ["curses"]},
    }
    llm = FakeLLM([
        Reply(tool_calls=[call("bash", command="python3 -c 'import sys; sys.stderr.write(\"_curses.error\"); sys.exit(1)'")]),
        Reply(content="It crashed; fixed by compiling only."),
    ], review_handler(review))
    agent = Agent(settings=settings, llm=llm, store=store)
    trace = agent.run_turn("make a curses snake")
    assert trace.outcome == "recovered"
    assert len(trace.failures()) == 1
    assert any(a == "case+ curses-no-tty" for a in trace.memory_actions)
    saved = store.get_trace(trace.trace_id)
    assert "case+ curses-no-tty" in saved.memory_actions
    # The review prompt saw the actual error text.
    assert "_curses.error" in llm.json_calls[-1][1]


def test_retry_recalls_case_into_context(settings, store):
    case_ops.file_case(store, "curses needs a TTY", "curses apps crash when run through bash tool",
                       "**How to apply:** use py_compile", case_id="curses-no-tty")
    llm = FakeLLM([Reply(content="ok")], review_handler({"action": "none"}, selected=["curses-no-tty"]))
    agent = Agent(settings=settings, llm=llm, store=store)
    trace = agent.run_turn("make a curses snake")
    assert trace.loaded_case_ids == ["curses-no-tty"]
    sent = llm.chat_calls[0]
    assert "curses-no-tty: curses apps crash" in sent[0]["content"]     # index in system prompt
    assert "use py_compile" in sent[1]["content"]                       # body in the user turn
    assert store.get_case("curses-no-tty").uses == 1
    # no tool steps -> nothing to judge -> only the recall call was made
    assert len(llm.json_calls) == 1


def test_unrelated_task_loads_nothing(settings, store):
    case_ops.file_case(store, "curses", "curses apps crash in bash tool", "b", case_id="curses-no-tty")
    llm = FakeLLM([Reply(content="ok")], review_handler({"action": "none"}, selected=[]))
    trace = Agent(settings=settings, llm=llm, store=store).run_turn("what is 2+2")
    assert trace.loaded_case_ids == []
    assert "<recalled-cases>" not in llm.chat_calls[0][1]["content"]


def test_rule_added_then_loaded_next_turn_and_promoted(settings, store):
    first_review = {"analysis": "x", "action": "add_rule",
                    "rule": {"id": "verify-headless", "text": "Verify TUI/GUI programs headlessly; never run them directly."}}
    second_review = {"analysis": "rule followed", "action": "none", "rule_verdicts": {"verify-headless": "helped"}}
    reviews = [first_review, second_review]
    llm = FakeLLM(
        [Reply(tool_calls=[call("bash", command="exit 1")]), Reply(content="done"),
         Reply(tool_calls=[call("bash", command="true")]), Reply(content="done")],
        lambda s, u: reviews.pop(0) if s == REVIEW_SYSTEM else {"selected": []},
    )
    agent = Agent(settings=settings, llm=llm, store=store)
    agent.run_turn("make a game")
    hv = store.current_harness()
    assert hv.get("verify-headless").status == "trial"
    agent.reset()
    t2 = agent.run_turn("make a game")
    assert "verify-headless" in t2.rule_ids
    assert "[verify-headless] (trial) Verify TUI/GUI" in llm.chat_calls[2][0]["content"]
    assert store.current_harness().get("verify-headless").status == "active"


def test_flag_failure_forces_review(settings, store):
    llm = FakeLLM([Reply(content="Snake done.")],
                  review_handler({"analysis": "walls not handled", "action": "none"}))
    agent = Agent(settings=settings, llm=llm, store=store)
    agent.run_turn("build snake")
    assert llm.json_calls == []
    agent.flag_failure("snake passes through walls")
    assert "USER SAYS IT IS WRONG: snake passes through walls" in llm.json_calls[-1][1]
    assert store.get_trace(agent.last_trace.trace_id).outcome == "failed"


def test_memory_off_is_a_plain_agent(settings, store):
    case_ops.file_case(store, "c", "d", "b", case_id="c")
    llm = FakeLLM([Reply(tool_calls=[call("bash", command="exit 2")]), Reply(content="x")])
    agent = Agent(settings=settings.with_(memory=False), llm=llm, store=store)
    trace = agent.run_turn("task")
    assert llm.json_calls == []
    assert "Case library" not in llm.chat_calls[0][0]["content"]
    assert trace.outcome == "recovered"


def test_malformed_tool_call_is_nudged(settings, store):
    llm = FakeLLM([Reply(error="tool_use_failed: bad"), Reply(content="ok")])
    trace = Agent(settings=settings, llm=llm, store=store).run_turn("t")
    assert trace.outcome == "ok"
    assert "not valid JSON" in llm.chat_calls[1][-1]["content"]


def test_step_limit_aborts(settings, store):
    llm = FakeLLM([Reply(tool_calls=[call("bash", command="true")]) for _ in range(20)],
                  review_handler({"action": "none"}))
    trace = Agent(settings=settings.with_(max_steps=3), llm=llm, store=store).run_turn("loop forever")
    assert trace.outcome == "aborted" and trace.steps == 3


def _tool_exchange(i: int, size: int) -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "type": "function",
         "function": {"name": "write_file", "arguments": json.dumps({"path": "a.py", "content": "x" * size})}}]},
        {"role": "tool", "tool_call_id": f"t{i}", "content": "y" * size},
    ]


def test_compact_clears_old_tool_traffic_but_keeps_pairs():
    msgs = [{"role": "user", "content": "task"}]
    for i in range(6):
        msgs += _tool_exchange(i, 4000)
    before = context_tokens(msgs)
    freed = compact(msgs, budget=5000)
    assert freed > 0 and context_tokens(msgs) < before
    assert msgs[1]["tool_calls"][0]["function"]["arguments"].count("elided") == 1
    assert CLEARED in msgs[2]["content"]
    assert msgs[-1]["content"] == "y" * 4000            # recent output untouched
    ids = [tc["id"] for m in msgs if m.get("tool_calls") for tc in m["tool_calls"]]
    answered = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert ids == answered


def test_compact_drops_old_turns_when_needed():
    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"task {i} " + "z" * 8000})
        msgs.append({"role": "assistant", "content": "done"})
    compact(msgs, budget=3000)
    assert msgs[0]["content"].startswith("[earlier conversation trimmed")
    assert msgs[-2]["content"].startswith("task 3")


def test_repeated_identical_failure_gets_nudged(settings, store):
    same = lambda: Reply(tool_calls=[call("edit_file", path="a.py", old_string="x", new_string="x")])  # noqa: E731
    (settings.workdir / "a.py").write_text("x\n")
    llm = FakeLLM([Reply(tool_calls=[call("read_file", path="a.py")]), same(), same(), same(), Reply(content="giving up")],
                  review_handler({"action": "none"}))
    Agent(settings=settings, llm=llm, store=store).run_turn("t")
    last_user = [m for m in llm.chat_calls[-1] if m["role"] == "user"][-1]
    assert "same failing call three times" in last_user["content"]


def test_provider_abort_is_not_reviewed(settings, store):
    from hx.llm import LLMError

    def boom(messages):
        raise LLMError("service unavailable")

    llm = FakeLLM([boom], review_handler({"action": "add_rule", "rule": {"id": "junk", "text": "junk"}}))
    trace = Agent(settings=settings, llm=llm, store=store).run_turn("t")
    assert trace.outcome == "aborted" and trace.abort_reason == "model_error"
    assert llm.json_calls == []
    assert store.current_harness().rules == []
