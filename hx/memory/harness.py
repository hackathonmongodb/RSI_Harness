"""General harness: a short list of standing rules loaded on every turn.

Lifecycle (the light gate):
  new rule -> status "trial" (already loaded, marked as trial)
  next relevant turn where the failure it targets does NOT recur -> "active"
  relevant turns where it recurs anyway -> strikes; 2 strikes on trial / 3 on active -> "retired"
Counter updates are saved in place; status/text changes create a new harness version.
"""
from __future__ import annotations

from typing import Optional

from hx.memory.models import RULE_CAP, RULE_MAX_CHARS, HarnessVersion, Rule, now, slugify
from hx.memory.store import Store

TRIAL_STRIKES_TO_RETIRE = 2
ACTIVE_STRIKES_TO_RETIRE = 3


def render_rules(hv: HarnessVersion) -> str:
    lines = []
    for r in hv.live_rules():
        tag = " (trial)" if r.status == "trial" else ""
        lines.append(f"- [{r.rule_id}]{tag} {r.text}")
    return "\n".join(lines)


def add_rule(store: Store, rule_id: str, text: str, why: str = "", trace_id: Optional[str] = None) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "rule skipped: empty text"
    if len(text) > RULE_MAX_CHARS:
        text = text[: RULE_MAX_CHARS - 1].rstrip() + "…"
    hv = store.current_harness()
    rid = slugify(rule_id or text, 40)
    existing = hv.get(rid)
    if existing is not None and existing.status != "retired":
        return revise_rule(store, rid, text=text, why=why, trace_id=trace_id)
    live = hv.live_rules()
    if len(live) >= RULE_CAP:
        return f"rule skipped: harness is full ({RULE_CAP} rules); retire one first"
    rules = [r for r in hv.rules if r.rule_id != rid] + [
        Rule(rule_id=rid, text=text, why=why, status="trial", source_trace=trace_id)
    ]
    new = store.commit_harness(rules, change=f"add rule {rid} (trial)", reason=why, trace_id=trace_id)
    return f"rule+ {rid} (trial) -> harness v{new.version}"


def revise_rule(
    store: Store,
    rule_id: str,
    text: Optional[str] = None,
    why: str = "",
    retire: bool = False,
    trace_id: Optional[str] = None,
) -> str:
    hv = store.current_harness()
    rule = hv.get(rule_id)
    if rule is None or rule.status == "retired":
        return f"rule skipped: no live rule {rule_id}"
    rules = [r.model_copy() for r in hv.rules]
    target = next(r for r in rules if r.rule_id == rule_id)
    if retire:
        target.status = "retired"
        change = f"retire rule {rule_id}"
    else:
        new_text = " ".join((text or "").split())[:RULE_MAX_CHARS]
        if not new_text or new_text == target.text:
            return f"rule unchanged: {rule_id}"
        target.text = new_text
        target.status = "trial"       # revised text has to earn its place again
        target.strikes = 0
        change = f"revise rule {rule_id} (trial)"
    target.why = why or target.why
    target.updated_at = now()
    new = store.commit_harness(rules, change=change, reason=why, trace_id=trace_id)
    return f"rule{'-' if retire else '~'} {rule_id} -> harness v{new.version}"


def judge_rules(store: Store, verdicts: dict[str, str], trace_id: Optional[str] = None) -> list[str]:
    """Apply the review's per-rule verdicts: helped | violated | irrelevant."""
    if not verdicts:
        return []
    hv = store.current_harness()
    rules = [r.model_copy() for r in hv.rules]
    changes: list[str] = []
    touched = False
    for r in rules:
        v = (verdicts.get(r.rule_id) or "").lower()
        if r.status == "retired" or v not in {"helped", "violated"}:
            continue
        touched = True
        r.trials += 1
        r.updated_at = now()
        if v == "helped":
            r.helped += 1
            if r.status == "trial":
                r.status = "active"
                changes.append(f"promote rule {r.rule_id}")
        else:
            r.strikes += 1
            limit = TRIAL_STRIKES_TO_RETIRE if r.status == "trial" else ACTIVE_STRIKES_TO_RETIRE
            if r.strikes >= limit:
                r.status = "retired"
                changes.append(f"retire rule {r.rule_id} ({r.strikes} strikes)")
    if changes:
        new = store.commit_harness(rules, change="; ".join(changes), reason="gate verdicts", trace_id=trace_id)
        return [f"{c} -> harness v{new.version}" for c in changes]
    if touched:
        hv.rules = rules
        store.update_counters(hv)
    return []


def harness_markdown(hv: HarnessVersion) -> str:
    """HARNESS.md: the general rules as the agent sees them, plus why each exists."""
    lines = [f"# Harness v{hv.version}", "", f"_Last change: {hv.change}_", ""]
    live = hv.live_rules()
    if not live:
        lines.append("(no rules yet — the agent adds one when a failure teaches a general lesson)")
    for r in live:
        tag = " (trial)" if r.status == "trial" else ""
        lines.append(f"- **[{r.rule_id}]**{tag} {r.text}")
        if r.why:
            lines.append(f"  - _why:_ {r.why}")
        lines.append(f"  - status `{r.status}` · helped {r.helped} · strikes {r.strikes}")
    return "\n".join(lines) + "\n"
