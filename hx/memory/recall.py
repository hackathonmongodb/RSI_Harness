"""Selective load: a cheap model picks <= 5 cases from their one-line descriptions, or none.

Prompt adapted from Claude Code's findRelevantMemories: be selective, skip if unsure.
"""
from __future__ import annotations

from typing import Iterable, Optional

from hx.memory.cases import render_index
from hx.memory.models import Case

MAX_SELECTED = 5

SELECT_SYSTEM = """You select past incident notes ("cases") that will help a coding agent with the user's request.
You get the request and a list of cases as `id: one-line description`.

Return the ids of cases that will CLEARLY help with this request (at most 5).
- Only pick a case if you are confident from its description that it applies to this request
  (same tool, library, file type, environment, or failure mode). Similar wording alone is not enough.
- If unsure, do not pick it. An empty list is a good answer.
Return JSON only: {"selected": ["case-id", ...]}"""


def select_cases(
    llm,
    query: str,
    cases: list[Case],
    *,
    model: Optional[str] = None,
    exclude: Iterable[str] = (),
) -> list[str]:
    skip = set(exclude)
    pool = [c for c in cases if c.status == "active" and c.case_id not in skip]
    if not pool or not (query or "").strip():
        return []
    user = f"Request:\n{query.strip()[:2000]}\n\nCases:\n{render_index(pool)}"
    try:
        data = llm.json(SELECT_SYSTEM, user, model=model, max_tokens=600)
    except Exception:  # noqa: BLE001 — recall failing must never block the turn
        return []
    raw = data.get("selected") or data.get("ids") or []
    if isinstance(raw, str):
        raw = [raw]
    valid = {c.case_id for c in pool}
    out: list[str] = []
    for cid in raw:
        cid = str(cid).strip().strip("`")
        if cid in valid and cid not in out:
            out.append(cid)
    return out[:MAX_SELECTED]
