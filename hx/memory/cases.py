"""Case library: one doc per incident, surfaced by its one-line description only."""
from __future__ import annotations

from typing import Iterable, Optional

from hx.memory.models import CASE_DESC_MAX, Case, slugify
from hx.memory.store import Store

INDEX_LIMIT = 60
CASE_BODY_MAX = 1500
MISLEADING_TO_RETIRE = 2


def render_index(cases: Iterable[Case], limit: int = INDEX_LIMIT) -> str:
    rows = [f"- {c.case_id}: {c.description}" for c in list(cases)[:limit]]
    return "\n".join(rows)


def render_case(case: Case) -> str:
    return f"### Case `{case.case_id}` — {case.title}\n{case.body.strip()}"


def _unique_id(store: Store, base: str) -> str:
    cid, n = base, 2
    while store.get_case(cid) is not None:
        cid = f"{base}-{n}"
        n += 1
    return cid


def file_case(
    store: Store,
    title: str,
    description: str,
    body: str,
    tags: Optional[list[str]] = None,
    case_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> str:
    title = " ".join((title or "").split())
    description = " ".join((description or "").split())[:CASE_DESC_MAX]
    body = (body or "").strip()[:CASE_BODY_MAX]
    if not description or not body:
        return "case skipped: needs a description and a body"
    cid = _unique_id(store, slugify(case_id or title or description))
    store.save_case(
        Case(
            case_id=cid,
            title=title or description,
            description=description,
            body=body,
            tags=[str(t)[:30] for t in (tags or [])][:6],
            source_trace=trace_id,
        )
    )
    return f"case+ {cid}"


def update_case(
    store: Store,
    case_id: str,
    description: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[list[str]] = None,
    retire: bool = False,
) -> str:
    case = store.get_case(case_id)
    if case is None:
        return f"case skipped: no case {case_id}"
    if retire:
        case.status = "retired"
    if description:
        case.description = " ".join(description.split())[:CASE_DESC_MAX]
    if body:
        case.body = body.strip()[:CASE_BODY_MAX]
    if tags:
        case.tags = [str(t)[:30] for t in tags][:6]
    store.save_case(case)
    return f"case{'-' if retire else '~'} {case_id}"


def mark_used(store: Store, case_ids: Iterable[str]) -> None:
    for cid in case_ids:
        c = store.get_case(cid)
        if c is not None:
            c.uses += 1
            store.save_case(c)


def judge_cases(store: Store, verdicts: dict[str, str]) -> list[str]:
    out = []
    for cid, verdict in (verdicts or {}).items():
        v = (verdict or "").lower()
        if v not in {"helped", "misleading"}:
            continue
        c = store.get_case(cid)
        if c is None or c.status != "active":
            continue
        if v == "helped":
            c.helped += 1
        else:
            c.misleading += 1
            if c.misleading >= MISLEADING_TO_RETIRE:
                c.status = "retired"
                out.append(f"case- {cid} (misleading x{c.misleading})")
        store.save_case(c)
    return out


def case_markdown(case: Case) -> str:
    """The case as a markdown file with frontmatter (exactly what the local store writes to disk)."""
    from hx.memory.store import _dump_front

    return _dump_front(case)
