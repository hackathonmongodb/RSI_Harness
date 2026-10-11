"""Harness data + pure assemble / apply_edits. No Atlas."""

from __future__ import annotations

from copy import deepcopy

from hx.models import AssembledPrompt, ContextComponent, Edit, Harness, Policy, Task

PREAMBLE = ContextComponent(
    id="preamble",
    title="Role",
    text=(
        "You are a MongoDB query generator. Given a natural-language request, "
        "return ONLY JSON (no markdown) with keys collection, operation "
        "('find' or 'aggregate'), filter, projection, sort, limit, pipeline. "
        "Read-only: never $out, $merge, insert, update, or delete."
    ),
)

TIPS = ContextComponent(
    id="query_tips",
    title="Query tips",
    text=(
        "Prefer find over aggregate unless grouping or multi-stage transforms "
        "are required. Put $match first. Use $project last. Never emit _id: null. "
        "Never use an empty new Date(); always use an explicit ISO date string."
    ),
)

# Placeholders until A introspects sample_mflix.movies on Atlas.
SCHEMA = ContextComponent(
    id="schema",
    title="Annotated schema (movies)",
    text=(
        "Movie { _id: ObjectId, title: string, year: number, plot: string, "
        "genres: string[], runtime: number, num_mflix_comments: number, "
        "released: Date, directors: string[], cast: string[], "
        "imdb: { rating: number, votes: number, id: number }, type: string }"
    ),
)

SAMPLES = ContextComponent(
    id="samples",
    title="Sample documents",
    text=(
        '[{"title":"The Martian","year":2015,"genres":["Adventure","Drama","Sci-Fi"],'
        '"runtime":144,"num_mflix_comments":12},'
        '{"title":"Spotlight","year":2015,"genres":["Crime","Drama"],'
        '"runtime":129,"num_mflix_comments":8}]'
    ),
)

INDEXES = ContextComponent(
    id="indexes",
    title="Indexes",
    text='[{"key":{"_id":1}},{"key":{"year":1}},{"key":{"title":1}}]',
)

H0 = Harness(
    harness_id="h0",
    version=1,
    rules=[
        "Return valid JSON only.",
        "Do not invent field names that are not implied by the question.",
    ],
    policy=Policy(
        max_retries=1,
        temperature=0.0,
        include_schema=False,
        include_samples=False,
        include_indexes=False,
    ),
    components=[PREAMBLE],
    mutation_reason="baseline: no schema/samples/indexes",
)

H_MONGODB = Harness(
    harness_id="h_mongodb",
    version=1,
    rules=[
        "Return valid JSON only.",
        "Do not invent field names; use the provided schema.",
        "Use indexes when filtering on year or title.",
        "Follow MongoDB natural-language-to-query prompting: schema, samples, indexes.",
    ],
    policy=Policy(
        max_retries=2,
        temperature=0.0,
        include_schema=True,
        include_samples=True,
        include_indexes=True,
    ),
    components=[PREAMBLE, TIPS, SCHEMA, SAMPLES, INDEXES],
    mutation_reason="mongodb nl-to-query harness",
)


def assemble(harness: Harness, task: Task | str) -> AssembledPrompt:
    nl = task.nl if isinstance(task, Task) else task
    enabled_ids: list[str] = []
    parts: list[str] = []

    policy = harness.policy
    for comp in harness.components:
        if not comp.enabled:
            continue
        if comp.id == "schema" and not policy.include_schema:
            continue
        if comp.id == "samples" and not policy.include_samples:
            continue
        if comp.id == "indexes" and not policy.include_indexes:
            continue
        enabled_ids.append(comp.id)
        parts.append(f"## {comp.title}\n{comp.text}")

    rules = "\n".join(f"- {r}" for r in harness.rules)
    system = (
        f"Harness: {harness.harness_id} v{harness.version}\n\n"
        + "\n\n".join(parts)
        + "\n\n## Rules\n"
        + (rules or "- (none)")
        + f"\n\n## Policy\nmax_retries={policy.max_retries} temperature={policy.temperature}"
    )
    user = (
        "Natural language query:\n"
        f"{nl}\n\n"
        "Respond with JSON only."
    )
    return AssembledPrompt(
        system=system,
        user=user,
        messages=[{"role": "user", "content": user}],
        context_components=enabled_ids,
    )


def apply_edits(harness: Harness, edits: list[Edit]) -> Harness:
    """Pure. delete_rule removes that exact string. set_policy touches one knob."""
    nxt = Harness.model_validate(deepcopy(harness.model_dump()))
    for edit in edits:
        if edit.op == "delete_rule":
            text = str(edit.value)
            nxt.rules = [r for r in nxt.rules if r != text]
        elif edit.op == "add_rule":
            text = str(edit.value)
            if text not in nxt.rules:
                nxt.rules.append(text)
        elif edit.op == "set_policy":
            if not edit.knob:
                raise ValueError("set_policy requires knob")
            if edit.knob not in Policy.model_fields:
                raise ValueError(f"unknown policy knob {edit.knob}")
            nxt.policy = nxt.policy.model_copy(update={edit.knob: edit.value})
        elif edit.op == "set_component_enabled":
            if not edit.component_id:
                raise ValueError("set_component_enabled requires component_id")
            found = False
            comps = []
            for c in nxt.components:
                if c.id == edit.component_id:
                    comps.append(c.model_copy(update={"enabled": bool(edit.value)}))
                    found = True
                else:
                    comps.append(c)
            if not found:
                raise ValueError(f"unknown component {edit.component_id}")
            nxt.components = comps
        else:
            raise ValueError(edit.op)
    nxt.parent_version = harness.version
    nxt.version = harness.version + 1
    return nxt
