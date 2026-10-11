"""The inner loop.

  user asks -> recall (0..5 cases) -> assemble (base + rules + case index) -> model <-> tools
            -> trace -> if something failed: self-review -> maybe one rule or one case

Recursive improvement = the next turn starts from a slightly better environment.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from hx import prompts, tools
from hx.config import Settings
from hx.llm import LLMError
from hx.memory import cases as case_ops
from hx.memory import recall, reflect
from hx.memory.models import Case, HarnessVersion, ToolEvent, TurnTrace, new_id
from hx.memory.store import Store
from hx.ui import UI

KEEP_RECENT = 6          # messages at the tail that compaction never touches
CLEARED = "[old output cleared to save context; re-run the tool if you need it]"
TRIMMED = "[earlier conversation trimmed to save context]"
MALFORMED_NUDGE = (
    "Your last tool call could not be parsed: its arguments were not valid JSON. Send it again. "
    "Escape double quotes, backslashes and newlines inside JSON strings. For a long file, write a short "
    "first version with write_file and extend it with edit_file."
)
LENGTH_NUDGE = "You ran out of output tokens. Continue, more concisely."
REPEAT_NUDGE = (
    "You have made the same failing call three times in a row. Stop repeating it: re-read the file or "
    "the error, then take a different approach, or finish and explain what is blocking you."
)


# ---------------------------------------------------------------------------- compaction
def _msg_tokens(m: dict[str, Any]) -> int:
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        n += len(tc["function"].get("arguments") or "")
    return n // 4 + 4


def context_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_msg_tokens(m) for m in messages)


def compact(messages: list[dict[str, Any]], budget: int, system_tokens: int = 0) -> int:
    """Shrink old tool traffic in place until the conversation fits the budget. Returns tokens freed.

    1. clear old tool outputs, 2. elide file bodies in old write/edit calls,
    3. drop whole earlier user turns (keeping tool_call/tool pairs intact).
    """
    before = context_tokens(messages)
    if before + system_tokens <= budget:
        return 0
    cut = max(0, len(messages) - KEEP_RECENT)
    for m in messages[:cut]:
        if m.get("role") == "tool" and len(m.get("content") or "") > 200:
            first = (m["content"].splitlines() or [""])[0][:100]
            m["content"] = f"{first}\n{CLEARED}"
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if fn.get("name") in ("write_file", "edit_file") and len(fn.get("arguments") or "") > 300:
                try:
                    args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    args = {}
                for k in ("content", "old_string", "new_string"):
                    if isinstance(args.get(k), str) and len(args[k]) > 80:
                        args[k] = f"[{len(args[k])} chars elided]"
                fn["arguments"] = json.dumps(args)
        if context_tokens(messages) + system_tokens <= budget:
            return before - context_tokens(messages)
    # Still too big: drop the oldest exchanges. Cutting right before a user message never splits
    # an assistant tool_call from its tool results.
    while context_tokens(messages) + system_tokens > budget:
        starts = [i for i, m in enumerate(messages) if m.get("role") == "user" and m.get("content") != TRIMMED]
        if len(starts) < 2:
            break
        del messages[: starts[1]]
        messages.insert(0, {"role": "user", "content": TRIMMED})
    return before - context_tokens(messages)


def _clip_args(args: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > limit:
            v = v[:limit] + f"\n... [{len(v) - limit} more chars]"
        out[k] = v
    return out


# ---------------------------------------------------------------------------- agent
@dataclass
class Agent:
    settings: Settings
    llm: Any
    store: Store
    ui: UI = field(default_factory=UI)
    session_id: str = field(default_factory=new_id)
    messages: list[dict[str, Any]] = field(default_factory=list)
    surfaced: set[str] = field(default_factory=set)
    last_trace: Optional[TurnTrace] = None
    cancel: threading.Event = field(default_factory=threading.Event)   # set by a UI to stop the turn
    ctx: tools.ToolContext = field(init=False)

    def __post_init__(self) -> None:
        self.ctx = tools.ToolContext(
            workdir=self.settings.workdir.resolve(),
            auto_approve=self.settings.auto_approve,
            bash_timeout=self.settings.bash_timeout,
            confirm=self.ui.confirm,
            load_case=self._load_case if self.settings.memory else None,
        )

    # -------------------------------------------------------------- helpers
    def _load_case(self, case_id: str) -> Optional[str]:
        c = self.store.get_case(case_id)
        if c is None or c.status != "active":
            return None
        if case_id not in self.surfaced:
            self.surfaced.add(case_id)
            case_ops.mark_used(self.store, [case_id])
            if self.last_trace is not None and case_id not in self.last_trace.loaded_case_ids:
                self.last_trace.loaded_case_ids.append(case_id)
        return case_ops.render_case(c)

    def reset(self) -> None:
        """Forget the conversation (not the memory). Used by /clear and /retry."""
        self.messages.clear()
        self.surfaced.clear()
        self.ctx.read_state.clear()

    def set_memory(self, on: bool) -> None:
        """Toggle recall + reflection (for with/without comparisons). Stored memory is untouched."""
        self.settings = self.settings.with_(memory=on)
        self.ctx.load_case = self._load_case if on else None

    def harness(self) -> HarnessVersion:
        if not self.settings.memory:
            return HarnessVersion(version=0)
        return self.store.current_harness()

    # -------------------------------------------------------------- one turn
    def run_turn(self, task: str) -> TurnTrace:
        memory = self.settings.memory
        self.cancel.clear()
        ev = self.ui.event
        hv = self.harness()
        all_cases: list[Case] = self.store.list_cases() if memory else []
        turn_t0 = time.perf_counter()
        ev("turn_start", task=task, memory=memory, harness_version=hv.version)

        # 1. recall: pick 0..5 cases by their one-line descriptions
        selected_ids: list[str] = []
        if memory and all_cases:
            pool = [c for c in all_cases if c.case_id not in self.surfaced]
            ev("recall_start", model=self.settings.fast_model,
               candidates=[{"id": c.case_id, "description": c.description} for c in pool],
               already_surfaced=sorted(self.surfaced))
            t0 = time.perf_counter()
            with self.ui.status("recalling cases"):
                selected_ids = recall.select_cases(
                    self.llm, task, all_cases, model=self.settings.fast_model, exclude=self.surfaced
                )
            ev("recall_done", selected=selected_ids, n_candidates=len(pool),
               latency_ms=int((time.perf_counter() - t0) * 1000))
        else:
            ev("recall_skip", reason="memory off" if not memory else "case library is empty")
        selected = [c for c in all_cases if c.case_id in selected_ids]
        if selected:
            self.surfaced.update(selected_ids)
            case_ops.mark_used(self.store, selected_ids)

        # 2. assemble: pinned base + harness rules + case index (+ selected bodies in the user turn)
        system = prompts.system_prompt(self.ctx.workdir, hv, all_cases, memory)
        sys_tokens = prompts.estimate_tokens(system)
        trace = TurnTrace(
            task=task,
            session_id=self.session_id,
            workdir=str(self.ctx.workdir),
            harness_version=hv.version,
            rule_ids=[r.rule_id for r in hv.live_rules()],
            loaded_case_ids=list(selected_ids),
            system_tokens=sys_tokens,
        )
        self.last_trace = trace
        self.ui.context(hv, len(all_cases), selected, sys_tokens, memory)
        user_msg = prompts.user_message(task, selected)
        ev("assemble", trace_id=trace.trace_id, harness_version=hv.version,
           rules=[{"id": r.rule_id, "status": r.status, "text": r.text} for r in hv.live_rules()],
           n_case_index=len(all_cases), loaded_cases=selected_ids, system_tokens=sys_tokens,
           user_tokens=prompts.estimate_tokens(user_msg), system_prompt=system, user_message=user_msg,
           history_messages=len(self.messages))

        self.messages.append({"role": "user", "content": user_msg})
        schemas = tools.schemas(memory)
        nudges = repeats = 0
        last_sig = None
        try:
            for step in range(1, self.settings.max_steps + 1):
                if self.cancel.is_set():
                    raise KeyboardInterrupt
                freed = compact(self.messages, self.settings.context_budget, sys_tokens)
                if freed:
                    ev("compact", freed_tokens=freed, context_tokens=context_tokens(self.messages) + sys_tokens,
                       budget=self.settings.context_budget)
                ctx_tok = context_tokens(self.messages) + sys_tokens
                ev("llm_start", step=step, model=self.settings.model, context_tokens=ctx_tok,
                   n_messages=len(self.messages) + 1)
                t0 = time.perf_counter()
                with self.ui.status("thinking"):
                    reply = self.llm.chat([{"role": "system", "content": system}] + self.messages, schemas)
                trace.prompt_tokens += reply.prompt_tokens
                trace.completion_tokens += reply.completion_tokens
                ev("llm_end", step=step, latency_ms=int((time.perf_counter() - t0) * 1000),
                   prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                   finish_reason=reply.finish_reason, content=reply.content, error=reply.error,
                   tool_calls=[{"name": tc.name, "summary": tools.summarize_args(tc.name, tc.arguments)}
                               for tc in reply.tool_calls])
                if reply.error:
                    nudges += 1
                    if nudges > 3:
                        trace.outcome, trace.abort_reason = "aborted", "malformed_output"
                        trace.final_text = "Stopped: the model kept producing malformed tool calls."
                        self.ui.notice(trace.final_text)
                        break
                    self.messages.append({"role": "user", "content": MALFORMED_NUDGE})
                    ev("nudge", reason="malformed tool call", text=MALFORMED_NUDGE)
                    continue
                if not reply.tool_calls:
                    if not reply.content and reply.finish_reason == "length" and nudges < 3:
                        nudges += 1
                        self.messages.append({"role": "user", "content": LENGTH_NUDGE})
                        ev("nudge", reason="ran out of output tokens", text=LENGTH_NUDGE)
                        continue
                    trace.final_text = reply.content
                    self.messages.append({"role": "assistant", "content": reply.content or "(done)"})
                    self.ui.final(reply.content)
                    break
                if reply.content:
                    self.ui.thought(reply.content)
                self.messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)}}
                        for tc in reply.tool_calls
                    ],
                })
                for tc in reply.tool_calls:
                    if self.cancel.is_set():
                        raise KeyboardInterrupt
                    trace.steps += 1
                    summary = tools.summarize_args(tc.name, tc.arguments)
                    self.ui.tool_start(tc.name, summary)
                    ev("tool_start", step=trace.steps, name=tc.name, summary=summary, args=_clip_args(tc.arguments))
                    t0 = time.perf_counter()
                    result = tools.execute(self.ctx, tc.name, tc.arguments)
                    benign = tools.is_benign(tc.name, tc.arguments, result)
                    self.ui.tool_end(tc.name, result, benign)
                    ev("tool_end", step=trace.steps, name=tc.name, summary=summary, ok=result.ok, kind=result.kind,
                       benign=benign, failure=not result.ok and not benign, output=result.output[:6000],
                       latency_ms=int((time.perf_counter() - t0) * 1000))
                    trace.events.append(ToolEvent(
                        step=trace.steps, tool=tc.name, args=summary, ok=result.ok, kind=result.kind,
                        output=(result.output[:700] if not result.ok else result.output[:160]), benign=benign,
                    ))
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.output or "(empty)"})
                    sig = (tc.name, json.dumps(tc.arguments, sort_keys=True), result.output[:300])
                    repeats = repeats + 1 if (not result.ok and sig == last_sig) else 0
                    last_sig = sig if not result.ok else None
                if repeats >= 2:
                    # Same call, same failure, three times: break the loop before it eats the step budget.
                    self.messages.append({"role": "user", "content": REPEAT_NUDGE})
                    ev("nudge", reason="same failing call 3x", text=REPEAT_NUDGE)
                    repeats = 0
            else:
                trace.outcome, trace.abort_reason = "aborted", "step_limit"
                trace.final_text = trace.final_text or f"Stopped after {self.settings.max_steps} steps."
                self.ui.notice(trace.final_text)
        except KeyboardInterrupt:
            trace.outcome, trace.abort_reason = "aborted", "interrupted"
            trace.final_text = "Interrupted by user."
            self._close_dangling_calls()
            self.ui.notice("interrupted")
        except LLMError as err:
            trace.outcome, trace.abort_reason = "aborted", "model_error"
            trace.final_text = f"Model error: {err}"
            self._close_dangling_calls()
            self.ui.error(trace.final_text)

        if trace.outcome == "ok" and trace.failures():
            trace.outcome = "recovered"
        self._persist(trace)
        ev("turn_end", trace_id=trace.trace_id, outcome=trace.outcome, abort_reason=trace.abort_reason,
           steps=trace.steps, failures=len(trace.failures()), prompt_tokens=trace.prompt_tokens,
           completion_tokens=trace.completion_tokens, final_text=trace.final_text,
           latency_ms=int((time.perf_counter() - turn_t0) * 1000))
        if memory:
            self.reflect(trace)
        ev("idle")
        return trace

    def _close_dangling_calls(self) -> None:
        """An interrupted turn can leave tool_calls without tool results; the API rejects that."""
        if not self.messages:
            return
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "(interrupted)"})
                break

    def _persist(self, trace: TurnTrace) -> None:
        try:
            self.store.save_trace(trace)
        except Exception as err:  # noqa: BLE001 — losing a trace must not lose the user's turn
            self.ui.error(f"could not save trace: {err}")

    # -------------------------------------------------------------- learning
    def reflect(self, trace: TurnTrace, force: bool = False) -> list[str]:
        ev = self.ui.event
        hv = self.store.current_harness()
        why = "forced" if force else reflect.review_reason(trace, hv)
        if not why:
            ev("review_skip", trace_id=trace.trace_id,
               reason="no real failure, no user feedback, nothing on trial to judge")
            return []
        all_cases = self.store.list_cases()
        review_input = reflect.build_review_input(trace, hv, all_cases)
        ev("review_start", trace_id=trace.trace_id, why=why, model=self.settings.reflect_model,
           review_input=review_input, review_system=reflect.REVIEW_SYSTEM)
        t0 = time.perf_counter()
        with self.ui.status("reviewing this turn"):
            review = reflect.review_turn(self.llm, trace, hv, all_cases, model=self.settings.reflect_model)
        if review is None:
            self.ui.notice("self-review failed; nothing learned this turn")
            ev("review_failed", trace_id=trace.trace_id)
            return []
        ev("review_result", trace_id=trace.trace_id, latency_ms=int((time.perf_counter() - t0) * 1000),
           analysis=review.analysis, action=review.action, rule=review.rule, case=review.case,
           rule_verdicts=review.rule_verdicts, case_verdicts=review.case_verdicts)
        before = self.store.current_harness().version
        actions = reflect.apply_review(self.store, review, trace)
        trace.memory_actions = [f"analysis: {review.analysis}"] + actions if review.analysis else actions
        self._persist(trace)
        self.ui.memory(review.analysis, review.action, actions)
        after_hv = self.store.current_harness()
        words = {w for a in actions for w in a.replace("(", " ").split()}
        touched = [r.rule_id for r in after_hv.rules if r.rule_id in words]
        touched += [c.case_id for c in self.store.list_cases(include_retired=True) if c.case_id in words]
        ev("memory_write", trace_id=trace.trace_id, actions=actions, action=review.action,
           harness_before=before, harness_after=after_hv.version, touched=touched)
        return actions

    def flag_failure(self, note: str) -> list[str]:
        """User says the last result is wrong (a bug they saw, a failed run). Learn from it."""
        trace = self.last_trace
        if trace is None:
            self.ui.notice("no previous turn to flag")
            return []
        trace.user_feedback = note.strip() or "the result did not work"
        trace.outcome = "failed"
        self._persist(trace)
        return self.reflect(trace, force=True)
