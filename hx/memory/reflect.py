"""Post-turn self-review: the agent calls the model on its own trace to decide what to remember.

Runs only when there is something to learn from (a real failure, user feedback) or something
to judge (trial rules / loaded cases). At most ONE memory write per turn:
  add_rule | revise_rule | file_case | update_case | none
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from hx.memory import cases as case_ops
from hx.memory import harness as rule_ops
from hx.memory.models import RULE_CAP, RULE_MAX_CHARS, Case, HarnessVersion, TurnTrace
from hx.memory.store import Store

ACTIONS = {"none", "add_rule", "revise_rule", "file_case", "update_case"}
INFRA_ABORTS = {"interrupted", "model_error", "malformed_output"}

REVIEW_SYSTEM = f"""You are the self-review step of a terminal coding agent. The agent just finished one turn.
Decide whether this turn taught something that should change the agent's environment for NEXT time.

Facts about the agent's runtime (use them when attributing causes): its bash tool runs each command in a
fresh non-interactive shell in the working directory, with stdin closed, no TTY, no display, output captured,
and a timeout. Files are changed only through read_file / write_file / edit_file.

The agent has two memory buckets:
1. HARNESS RULES (general). Short standing rules loaded on EVERY turn, like a constitution.
   Add one only when the lesson applies far beyond this task (any project / any task of this kind).
   Imperative, <= {RULE_MAX_CHARS} chars. The harness must stay small (cap {RULE_CAP}); prefer revising an existing rule.
2. CASE LIBRARY (specific). One note per incident, loaded only when a future request looks related,
   selected by its one-line description. Use for tool-, library-, environment- or project-specific gotchas.

Choose exactly ONE action:
- "none": nothing reusable. Use it when the failure was a typo or one-off slip, the agent recovered by simply
  reading the error, the fix is now visible in the code itself, or it is an ephemeral task detail.
  Model/provider problems (malformed model output, rate limits, API errors) are infrastructure, not lessons.
  Do NOT store fix recipes already in the code, file paths, or anything secret. Most turns should be "none".
- "add_rule": a new general rule.
- "revise_rule": an existing rule was wrong, too narrow, or misleading -> new text (or retire it).
- "file_case": a new case.
- "update_case": an existing case (see index) covers this incident -> refine it instead of duplicating.

A case body is markdown with exactly these parts:
**Symptom:** what was observed (the actual error text, shortened)
**Root cause:** why it happened
**When it applies:** how to recognise the situation next time
**How to apply:** what to do instead
The case description is ONE specific line (<= 150 chars) naming the tool/library/situation, because a
selector will read only that line to decide whether to load the case.

Also judge what was loaded this turn — give a verdict for EVERY id under RULES TO JUDGE and
CASES LOADED THIS TURN (a case that steered the agent away from a past mistake "helped"):
- rule_verdicts, for each listed rule: "helped" (relevant here and the problem it targets did not happen),
  "violated" (relevant here but the problem it targets happened anyway), or "irrelevant".
- case_verdicts, for each loaded case: "helped", "misleading", or "irrelevant".

Return JSON only:
{{"analysis": "1-3 sentences: root cause, not symptom",
  "action": "none|add_rule|revise_rule|file_case|update_case",
  "rule": {{"id": "kebab-case-id", "text": "...", "why": "...", "retire": false}},
  "case": {{"id": "existing id for update_case, else a new kebab id", "title": "...", "description": "...",
           "body": "...", "tags": ["..."]}},
  "rule_verdicts": {{"rule-id": "helped|violated|irrelevant"}},
  "case_verdicts": {{"case-id": "helped|misleading|irrelevant"}}}}
Omit "rule"/"case" when not used."""


_SECRET = re.compile(
    r"(gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|mongodb(\+srv)?://[^\s:/]+:[^\s@]+@|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def redact(text: str) -> str:
    return _SECRET.sub("[REDACTED]", text or "")


@dataclass
class Review:
    analysis: str = ""
    action: str = "none"
    rule: dict[str, Any] = field(default_factory=dict)
    case: dict[str, Any] = field(default_factory=dict)
    rule_verdicts: dict[str, str] = field(default_factory=dict)
    case_verdicts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Review":
        action = str(data.get("action") or "none").strip().lower()
        return cls(
            analysis=str(data.get("analysis") or "")[:600],
            action=action if action in ACTIONS else "none",
            rule=data.get("rule") if isinstance(data.get("rule"), dict) else {},
            case=data.get("case") if isinstance(data.get("case"), dict) else {},
            rule_verdicts={str(k): str(v) for k, v in (data.get("rule_verdicts") or {}).items()}
            if isinstance(data.get("rule_verdicts"), dict) else {},
            case_verdicts={str(k): str(v) for k, v in (data.get("case_verdicts") or {}).items()}
            if isinstance(data.get("case_verdicts"), dict) else {},
        )


def review_reason(trace: TurnTrace, harness: HarnessVersion) -> Optional[str]:
    """Why this turn deserves a self-review, or None."""
    if trace.user_feedback:
        return "user flagged the result as wrong"
    fails = trace.failures()
    if fails:
        return f"{len(fails)} real failure(s) during the turn"
    if trace.outcome == "failed":
        return "turn failed"
    if trace.outcome == "aborted":
        # A provider hiccup or a ctrl-c is not a lesson about how to do the work.
        return None if trace.abort_reason in INFRA_ABORTS else f"turn aborted ({trace.abort_reason})"
    if trace.steps == 0:
        return None   # pure chat turn: nothing to judge
    trial = {r.rule_id for r in harness.live_rules() if r.status == "trial"} & set(trace.rule_ids)
    if trial:
        return "trial rule(s) to judge: " + ", ".join(sorted(trial))
    if trace.loaded_case_ids:
        return "loaded case(s) to judge: " + ", ".join(trace.loaded_case_ids)
    return None


def should_review(trace: TurnTrace, harness: HarnessVersion) -> bool:
    return review_reason(trace, harness) is not None


def render_trace(trace: TurnTrace, max_events: int = 30) -> str:
    rows = []
    events = trace.events
    if len(events) > max_events:
        # Keep every failure plus the start and end of the turn.
        keep = {e.step for e in events if not e.ok and not e.benign}
        keep |= {e.step for e in events[:8]} | {e.step for e in events[-8:]}
        events = [e for e in events if e.step in keep]
    for e in events:
        mark = "ok " if e.ok else ("~  " if e.benign else "ERR")
        line = f"{e.step:>2}. [{mark}] {e.tool}({e.args})"
        if not e.ok or e.output:
            excerpt = e.output if not e.ok else e.output[:120]
            line += "\n      -> " + excerpt.replace("\n", "\n         ")
        rows.append(line)
    parts = [
        f"TASK: {trace.task[:1500]}",
        f"OUTCOME: {trace.outcome}" + (f" ({trace.abort_reason})" if trace.abort_reason else "")
        + (f"  (USER SAYS IT IS WRONG: {trace.user_feedback})" if trace.user_feedback else ""),
        "ACTIONS:\n" + ("\n".join(rows) if rows else "(no tool calls)"),
        f"FINAL REPLY: {trace.final_text[:800] or '(none)'}",
    ]
    return redact("\n\n".join(parts))


def build_review_input(trace: TurnTrace, harness: HarnessVersion, all_cases: list[Case]) -> str:
    used_rules = [r for r in harness.live_rules() if r.rule_id in set(trace.rule_ids)]
    rules = "\n".join(
        f"- [{r.rule_id}] ({r.status}) {r.text}" for r in harness.live_rules()
    ) or "(none)"
    loaded = [c for c in all_cases if c.case_id in set(trace.loaded_case_ids)]
    return "\n\n".join(
        [
            f"CURRENT HARNESS RULES ({len(harness.live_rules())}/{RULE_CAP}):\n{rules}",
            "RULES TO JUDGE: " + (", ".join(r.rule_id for r in used_rules) or "(none)"),
            "CASE INDEX:\n" + (case_ops.render_index(all_cases) or "(empty)"),
            "CASES LOADED THIS TURN: " + (", ".join(c.case_id for c in loaded) or "(none)"),
            "TURN TRACE:\n" + render_trace(trace),
        ]
    )


def review_turn(llm, trace: TurnTrace, harness: HarnessVersion, all_cases: list[Case], *, model: Optional[str] = None) -> Optional[Review]:
    try:
        data = llm.json(REVIEW_SYSTEM, build_review_input(trace, harness, all_cases), model=model, max_tokens=2000)
    except Exception:  # noqa: BLE001 — a failed review only means nothing is learned this turn
        return None
    return Review.from_json(data)


def apply_review(store: Store, review: Review, trace: TurnTrace) -> list[str]:
    """Write at most one memory item, then apply gate verdicts. Returns human-readable actions."""
    out: list[str] = []
    a = review.action
    if a in ("add_rule", "revise_rule"):
        r = review.rule
        text = redact(str(r.get("text") or ""))
        if a == "add_rule":
            out.append(rule_ops.add_rule(store, str(r.get("id") or ""), text, str(r.get("why") or review.analysis), trace.trace_id))
        else:
            out.append(rule_ops.revise_rule(store, str(r.get("id") or ""), text=text, why=str(r.get("why") or review.analysis),
                                            retire=bool(r.get("retire")), trace_id=trace.trace_id))
    elif a in ("file_case", "update_case"):
        c = review.case
        body = redact(str(c.get("body") or ""))
        desc = redact(str(c.get("description") or ""))
        cid = str(c.get("id") or "").strip()
        if a == "update_case" and cid and store.get_case(cid) is not None:
            out.append(case_ops.update_case(store, cid, description=desc or None, body=body or None, tags=c.get("tags") or None))
        else:
            out.append(case_ops.file_case(store, str(c.get("title") or ""), desc, body, c.get("tags") or [],
                                          case_id=cid or None, trace_id=trace.trace_id))
    # Only judge rules/cases that were actually in context this turn.
    rv = {k: v for k, v in review.rule_verdicts.items() if k in set(trace.rule_ids)}
    out.extend(rule_ops.judge_rules(store, rv, trace.trace_id))
    cv = {k: v for k, v in review.case_verdicts.items() if k in set(trace.loaded_case_ids)}
    out.extend(case_ops.judge_cases(store, cv))
    return [s for s in out if s]
