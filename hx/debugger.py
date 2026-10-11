"""Failure attribution. Public signature matches SPEC.md section 7.

The LLM never receives a MongoClient. analyze() only returns reports;
the orchestrator (hx.loop) persists them.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from hx.models import Component, Report, Trace

# Tests patch this. Production may later load traces via orchestrator-injected data.
_TRACE_LOADER = None


def _prefer_mocks() -> bool:
    return os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}


def _complete(model: str, system: str, messages: list[dict], **kwargs):
    if _prefer_mocks():
        from tests.mocks import complete as _c
        return _c(model, system, messages, **kwargs)
    from hx.llm import complete as _c
    return _c(model, system, messages, **kwargs)


def _load_traces(harness_id: str, trace_ids: list[str]) -> list[Trace]:
    if _TRACE_LOADER is not None:
        return _TRACE_LOADER(harness_id, trace_ids)
    from tests.mocks import get_traces

    return get_traces(harness_id, trace_ids)


def _load_components(harness_id: str) -> list[Component]:
    if _prefer_mocks():
        from tests.mocks import load_harness
    else:
        from hx.harness import load_harness
    _, comps = load_harness(harness_id)
    return comps


def _load_question(task_id: str) -> str:
    """Train task question only — never gold pipelines."""
    try:
        from tests.mocks import get_task

        t = get_task(task_id)
        if t is None:
            return ""
        if t.split == "heldout":
            return ""
        return t.question
    except ImportError:
        return ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return m.group(1) if m else text


def _parse_report(trace: Trace, payload: dict[str, Any]) -> Report:
    missing = bool(payload.get("missing_component"))
    implicated = payload.get("implicated_component_id")
    if missing:
        implicated = None
    return Report(
        trace_id=trace.trace_id,
        task_id=trace.task_id,
        harness_id=trace.harness_id,
        outcome="pass" if trace.score >= 1.0 else "fail",
        root_cause=str(payload.get("root_cause") or "unspecified"),
        implicated_component_id=implicated,
        implicated_kind=payload.get("implicated_kind"),
        missing_component=missing,
        confidence=float(payload.get("confidence") or 0.5),
        evidence=str(payload.get("evidence") or (trace.error or "")[:400]),
    )


DEBUGGER_SYSTEM = """You attribute a train-split query-agent failure to a harness component.
Return ONLY JSON:
{"root_cause": str, "implicated_component_id": str|null, "implicated_kind": "rule"|"policy"|"tool"|null,
 "missing_component": bool, "confidence": float, "evidence": str}
If something should be ADDED, set missing_component=true and implicated_component_id=null.
Use the error text as the execution diff. Do not invent gold pipelines or held-out tasks.
"""


def _prompt_messages(trace: Trace, comps: list[Component], question: str) -> list[dict]:
    body = {
        "question": question,
        "components": [
            {
                "component_id": c.component_id,
                "kind": c.kind,
                "pinned": c.pinned,
                "content": c.content,
                "value": c.value,
            }
            for c in comps
        ],
        "raw_output": trace.raw_output,
        "error_type": trace.error_type,
        "error": trace.error,
        "score": trace.score,
    }
    return [{"role": "user", "content": json.dumps(body)}]


def _analyze_one(trace: Trace, comps: list[Component], question: str) -> Optional[Report]:
    model = os.environ.get(
        "META_MODEL",
        os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    messages = _prompt_messages(trace, comps, question)
    blob = json.dumps(messages)
    if "HELD OUT" in blob or "heldout_outcomes" in blob or "gold" in blob.lower() and "pipeline" in blob.lower():
        # Isolation: never send held-out or gold. Rebuild without question if tainted.
        messages = _prompt_messages(trace, comps, "")
    text = _complete(model, DEBUGGER_SYSTEM, messages).text
    try:
        payload = json.loads(_strip_fences(text))
        return _parse_report(trace, payload)
    except Exception:
        text = _complete(model, DEBUGGER_SYSTEM + " Retry: return valid JSON only.", messages).text
        try:
            payload = json.loads(_strip_fences(text))
            return _parse_report(trace, payload)
        except Exception:
            return None


def analyze(harness_id: str, trace_ids: list[str]) -> list[Report]:
    traces = _load_traces(harness_id, trace_ids)
    comps = _load_components(harness_id)
    reports: list[Report] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(_analyze_one, tr, comps, _load_question(tr.task_id))
            for tr in traces
        ]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                reports.append(r)
    reports.sort(key=lambda r: r.trace_id)
    return reports
