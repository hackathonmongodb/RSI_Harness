"""Harness load / assemble / apply_edits. SPEC §4 and §7. Atlas writes when A’s client works."""

from __future__ import annotations

import copy
import uuid
from typing import Any

from hx.models import Component, ComponentRef, Edit, Harness, Phase, Task

# In-memory store so C and tests run before Atlas is up.
_COMPONENTS: dict[tuple[str, int], Component] = {}
_HARNESSES: dict[str, Harness] = {}
_LATEST: dict[str, int] = {}

def _token_cost(text: str | None) -> int:
    return len(text or "") // 4


RULE_BASE = (
    "You generate MongoDB aggregation pipelines from natural language. "
    "Return ONLY JSON: {\"collection\": \"<name>\", \"pipeline\": [ ... ]} "
    "in Extended JSON. Aggregation only. Never $out, $merge, $function, "
    "$accumulator, or $where. Max 12 stages."
)

# PROMPTS B: rule.tip_1..9 = MongoDB's nine general tips (AGENTS.md).
TIPS = [
    ("rule.tip_1", "Use proper operators and types (ObjectId, ISODate); never loose JS types."),
    ("rule.tip_2", "Use an aggregation pipeline for complex queries ($match, $group, $lookup)."),
    ("rule.tip_3", "Prefer indexed fields; avoid $where and collection scans."),
    ("rule.tip_4", "Add $sort and $limit when the question asks for order or top-N."),
    ("rule.tip_5", "Handle null / missing / empty arrays with $exists and $type."),
    ("rule.tip_6", "Never emit _id: null in result objects or $group identities."),
    ("rule.tip_7", "Never use new Date() with no argument; always give an explicit date."),
    ("rule.tip_8", "For Decimal128, prefer range queries over exact equality."),
    ("rule.tip_9", "Use $elemMatch, $all, $size for array matching."),
]

# Train-style mflix questions only (not the held-out fallback tasks).
FEWSHOTS = [
    (
        "rule.fewshot_1",
        'Q: genres and runtime of 10 movies from 2015 with the most comments. Return genres, runtime. '
        'A: {"collection":"movies","pipeline":[{"$match":{"year":2015}},'
        '{"$sort":{"num_mflix_comments":-1}},{"$limit":10},'
        '{"$project":{"_id":0,"genres":1,"runtime":1}}]}',
    ),
    (
        "rule.fewshot_2",
        'Q: List title and year of 5 sci-fi movies from 2010. Return title, year. '
        'A: {"collection":"movies","pipeline":[{"$match":{"year":2010,"genres":"Sci-Fi"}},'
        '{"$limit":5},{"$project":{"_id":0,"title":1,"year":1}}]}',
    ),
    (
        "rule.fewshot_3",
        'Q: Movies released between 1990 and 2000, newest first, limit 10. Return title, year. '
        'A: {"collection":"movies","pipeline":[{"$match":{"year":{"$gte":1990,"$lte":2000}}},'
        '{"$sort":{"year":-1}},{"$limit":10},{"$project":{"_id":0,"title":1,"year":1}}]}',
    ),
]


def _put_component(c: Component) -> Component:
    _COMPONENTS[(c.component_id, c.version)] = c
    prev = _LATEST.get(c.component_id, 0)
    if c.version >= prev:
        _LATEST[c.component_id] = c.version
    return c


def _put_harness(h: Harness) -> Harness:
    h.n_rules = sum(
        1
        for r in h.components
        if _COMPONENTS.get((r.component_id, r.version)) and _COMPONENTS[(r.component_id, r.version)].kind == "rule"
    )
    h.n_tools = sum(
        1
        for r in h.components
        if _COMPONENTS.get((r.component_id, r.version)) and _COMPONENTS[(r.component_id, r.version)].kind == "tool"
    )
    _HARNESSES[h.harness_id] = h
    return h


def _policy(cid: str, name: str, value: Any, version: int = 1) -> Component:
    return Component(component_id=cid, version=version, kind="policy", name=name, value=value)


def _default_policies(
    *,
    cot: bool,
    n_sample: int,
    schema_style: str,
    indexes: bool,
    retries: int,
    version: int = 1,
) -> list[Component]:
    return [
        _policy("policy.n_sample_docs", "n_sample_docs", n_sample, version),
        _policy("policy.doc_truncate_chars", "doc_truncate_chars", 300, version),
        _policy("policy.schema_style", "schema_style", schema_style, version),
        _policy("policy.include_indexes", "include_indexes", indexes, version),
        _policy("policy.cot", "cot", cot, version),
        _policy("policy.max_retries", "max_retries", retries, version),
        _policy("policy.sample_docs_tool", "sample_docs_tool", False, version),
    ]


def _refs(comps: list[Component]) -> list[ComponentRef]:
    return [ComponentRef(component_id=c.component_id, version=c.version) for c in comps]


def seed_baselines(dataset: str = "sample_mflix") -> None:
    if "h0" in _HARNESSES:
        return
    base = Component(
        component_id="rule.base",
        version=1,
        kind="rule",
        name="base",
        content=RULE_BASE,
        pinned=True,
        token_cost=_token_cost(RULE_BASE),
    )
    h0_comps = [base, *_default_policies(cot=False, n_sample=0, schema_style="names", indexes=False, retries=0)]
    for c in h0_comps:
        _put_component(c)
    _put_harness(
        Harness(
            harness_id="h0",
            dataset=dataset,
            generation=0,
            phase="baseline",
            components=_refs(h0_comps),
            status="static",
        )
    )

    mongo_comps = [base]
    for cid, text in TIPS + FEWSHOTS:
        mongo_comps.append(
            Component(
                component_id=cid,
                version=1,
                kind="rule",
                name=cid,
                content=text,
                token_cost=_token_cost(text),
            )
        )
    mongo_comps.extend(
        _default_policies(
            cot=True,
            n_sample=3,
            schema_style="ts_annotated",
            indexes=True,
            retries=0,
            version=2,
        )
    )
    for c in mongo_comps:
        _put_component(c)
    _put_harness(
        Harness(
            harness_id="h_mongodb",
            dataset=dataset,
            generation=0,
            phase="baseline",
            components=_refs(mongo_comps),
            status="static",
        )
    )


seed_baselines()


def resolve_components(harness: Harness) -> list[Component]:
    out = []
    for ref in harness.components:
        c = _COMPONENTS.get((ref.component_id, ref.version))
        if c and c.status == "active":
            out.append(c)
    return out


def load_harness(harness_id: str) -> tuple[Harness, list[Component]]:
    h = _HARNESSES.get(harness_id)
    if h is None:
        try:
            from hx.config import get_client

            db = get_client("meta")["hx_harness"]
            doc = db.harnesses.find_one({"harness_id": harness_id})
            if not doc:
                raise KeyError(harness_id)
            h = Harness.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for ref in h.components:
                cdoc = db.components.find_one(
                    {"component_id": ref.component_id, "version": ref.version}
                )
                if cdoc:
                    _put_component(Component.model_validate({k: v for k, v in cdoc.items() if k != "_id"}))
            _put_harness(h)
        except Exception as err:
            raise KeyError(f"unknown harness {harness_id}: {err}") from err
    return h, resolve_components(h)


def _policy_map(comps: list[Component]) -> dict[str, Any]:
    return {c.component_id: c.value for c in comps if c.kind == "policy"}


def assemble(
    harness: Harness,
    comps: list[Component],
    task: Task,
    ctx: dict,
) -> tuple[str, list[dict], list[dict]]:
    """system, messages, tools. Prompt shape follows SPEC §4 + MongoDB NL-to-query docs."""
    policies = _policy_map(comps)
    rules = [c for c in comps if c.kind == "rule"]
    rules_sorted = sorted(rules, key=lambda c: c.component_id)
    sys_parts = [c.content or "" for c in rules_sorted if c.content]
    if policies.get("policy.cot"):
        sys_parts.append(
            "Think step by step about collections, operators, fields, indexes, types, and arrays, "
            "then output the JSON only."
        )
    system = "\n\n".join(p.strip() for p in sys_parts if p.strip())

    n_sample = int(policies.get("policy.n_sample_docs") or 0)
    trunc = int(policies.get("policy.doc_truncate_chars") or 300)
    style = policies.get("policy.schema_style") or "names"
    want_idx = bool(policies.get("policy.include_indexes"))

    db_name = task.dataset
    desc = ctx.get("description", db_name)
    user_parts = [
        f"Database: {db_name}",
        f"Description: {desc}",
    ]
    collections = dict(ctx.get("collections") or {})
    if task.collection and task.collection not in collections:
        collections[task.collection] = {}
    names = list(collections)
    if task.collection in names:
        names = [task.collection] + [n for n in names if n != task.collection]
    for cname in names:
        col = collections.get(cname) or {}
        user_parts.append(f"### Collection `{cname}`")
        if col.get("description"):
            user_parts.append(col["description"])
        if style == "ts_annotated" and col.get("ts_schema"):
            user_parts.append("Schema:\n" + str(col["ts_schema"]))
        elif col.get("ts_schema") or col.get("names"):
            user_parts.append("Schema (field names):\n" + str(col.get("names") or col.get("ts_schema")))
        if n_sample:
            docs = list(col.get("sample_docs") or [])[:n_sample]
            clipped = []
            for d in docs:
                s = d if isinstance(d, str) else str(d)
                clipped.append(s[:trunc])
            if clipped:
                user_parts.append("Example documents:\n" + "\n".join(clipped))
        if want_idx and col.get("indexes"):
            user_parts.append("Indexes:\n" + str(col["indexes"]))
    user_parts.append("Natural language query:\n" + task.question)
    user_parts.append("Respond with JSON only: {\"collection\": \"...\", \"pipeline\": [...]} ")
    user = "\n\n".join(user_parts)

    tools: list[dict] = []
    if policies.get("policy.sample_docs_tool"):
        tools.append(
            {
                "name": "sample_docs",
                "description": "Fetch n sample documents from a collection.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "n": {"type": "integer"},
                    },
                    "required": ["collection"],
                },
            }
        )
    messages = [{"role": "user", "content": user}]
    return system, messages, tools


def apply_edits(parent_id: str, edits: list[Edit], phase: Phase, generation: int) -> Harness:
    parent, comps = load_harness(parent_id)
    by_id = {c.component_id: c for c in comps}
    for edit in edits:
        current = by_id.get(edit.target_component_id)
        if current and current.pinned and edit.op in {"modify", "delete"}:
            raise ValueError(f"pinned component {edit.target_component_id} cannot be {edit.op}")
        if edit.op == "delete":
            by_id.pop(edit.target_component_id, None)
            continue
        if edit.op == "modify":
            if edit.kind == "policy" and current is None:
                raise ValueError("cannot add policy keys")
            if current is None:
                raise ValueError(f"modify missing {edit.target_component_id}")
            nxt = copy.deepcopy(edit.component or current)
            nxt.component_id = edit.target_component_id
            nxt.kind = edit.kind
            nxt.version = current.version + 1
            nxt.created_gen = generation
            if edit.kind == "policy":
                # only the value (and optional name) may change
                nxt.content = current.content
                nxt.template = current.template
                nxt.pinned = current.pinned
                if edit.component is not None and edit.component.value is not None:
                    nxt.value = edit.component.value
                nxt.name = current.name
            _put_component(nxt)
            by_id[nxt.component_id] = nxt
            continue
        if edit.op == "add":
            if edit.kind == "policy":
                raise ValueError("never add policy keys")
            nxt = copy.deepcopy(edit.component)
            if nxt is None:
                raise ValueError("add requires component")
            nxt.version = 1
            nxt.created_gen = generation
            _put_component(nxt)
            by_id[nxt.component_id] = nxt
    child_id = f"{parent.harness_id}-g{generation}-{uuid.uuid4().hex[:6]}"
    child = Harness(
        harness_id=child_id,
        parent_id=parent.harness_id,
        dataset=parent.dataset,
        generation=generation,
        phase=phase,
        components=_refs(list(by_id.values())),
        status="candidate",
    )
    child = _put_harness(child)
    _persist_apply(child, list(by_id.values()))
    return child


def _persist_apply(child: Harness, comps: list[Component]) -> None:
    """Write new component versions + child harness. Meta can write hx_harness (SPEC §2)."""
    try:
        from hx.config import get_client

        db = get_client("meta")["hx_harness"]
        for c in comps:
            db.components.update_one(
                {"component_id": c.component_id, "version": c.version},
                {"$set": c.model_dump()},
                upsert=True,
            )
        db.harnesses.update_one(
            {"harness_id": child.harness_id},
            {"$set": child.model_dump()},
            upsert=True,
        )
    except Exception:
        return


STUB_CTX = {
    "description": "Atlas sample_mflix movie catalog",
    "collections": {
        "movies": {
            "description": "Movie documents.",
            "ts_schema": (
                "interface Movie { _id: ObjectId; title: string; year: number; "
                "genres: string[]; runtime: number; num_mflix_comments: number; "
                "cast: string[]; directors: string[]; released: Date }"
            ),
            "names": "title year genres runtime num_mflix_comments cast directors released",
            "sample_docs": [
                '{"title":"The Martian","year":2015,"genres":["Adventure","Sci-Fi"],"runtime":144,"num_mflix_comments":12}',
                '{"title":"Spotlight","year":2015,"genres":["Crime","Drama"],"runtime":129,"num_mflix_comments":8}',
                '{"title":"Inception","year":2010,"genres":["Sci-Fi"],"runtime":148,"num_mflix_comments":20}',
            ],
            "indexes": '[{"key":{"year":1}},{"key":{"title":1}}]',
        }
    },
}
