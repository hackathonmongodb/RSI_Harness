"""Hand-labeled train failures for debugger attribution.

Each case: question, model output, error, expected implicated component
or missing_component=True. No gold pipelines, no held-out tasks.
"""
from __future__ import annotations

from hx.models import ComponentRef, Trace

CASES: list[dict] = [
    {
        "id": "decimal128",
        "question": "Average imdb.rating as avg_rating (Decimal128). Output fields: avg_rating.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{"awards.wins":{"$eq":1.0}}}]}',
        "error_type": "wrong_result",
        "error": "exact equality on Decimal128 missed rows",
        "implicated_component_id": "rule.types",
        "missing_component": False,
    },
    {
        "id": "missing_lookup",
        "question": "Join comments to movies; output fields: title, text.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{}}]}',
        "error_type": "wrong_result",
        "error": "no $lookup; comments never joined",
        "implicated_component_id": None,
        "missing_component": True,
    },
    {
        "id": "dates",
        "question": "Movies released after 2010-01-01; output fields: title, year.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{"released":"2010-01-01"}}]}',
        "error_type": "wrong_result",
        "error": "compared date field to a string, not ISODate",
        "implicated_component_id": "rule.types",
        "missing_component": False,
    },
    {
        "id": "output_fields",
        "question": "Return only title and year, limit 5. Output fields: title, year.",
        "raw_output": '{"collection":"movies","pipeline":[{"$limit":5}]}',
        "error_type": "wrong_result",
        "error": "extra fields in result objects",
        "implicated_component_id": "rule.base",
        "missing_component": False,
    },
    {
        "id": "arrays",
        "question": "Movies with both Action and Comedy in genres. Output fields: title, genres.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{"genres":"Action"}}]}',
        "error_type": "wrong_result",
        "error": "need $all/$elemMatch on genres array",
        "implicated_component_id": None,
        "missing_component": True,
    },
    {
        "id": "sample_docs_policy",
        "question": "Count theaters in Minnesota. Output fields: n.",
        "raw_output": '{"collection":"movies","pipeline":[{"$count":"n"}]}',
        "error_type": "wrong_result",
        "error": "wrong collection; no sample docs in prompt (n_sample_docs=0)",
        "implicated_component_id": "policy.n_sample_docs",
        "missing_component": False,
    },
    {
        "id": "cot_parse",
        "question": "Top 3 directors by count. Output fields: director, n. Order by n desc.",
        "raw_output": "Sure, here is a pipeline: db.movies.aggregate...",
        "error_type": "parse",
        "error": "not valid JSON / EJSON",
        "implicated_component_id": "policy.cot",
        "missing_component": False,
    },
    {
        "id": "nested",
        "question": "Filter imdb.rating greater than 8. Output fields: title, imdb.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{"rating":{"$gt":8}}}]}',
        "error_type": "wrong_result",
        "error": "nested field imdb.rating not used",
        "implicated_component_id": None,
        "missing_component": True,
    },
    {
        "id": "blocked_where",
        "question": "Find movies where title matches a JS predicate. Output fields: title.",
        "raw_output": '{"collection":"movies","pipeline":[{"$match":{"$where":"this.year>2000"}}]}',
        "error_type": "blocked",
        "error": "blocked stage $where",
        "implicated_component_id": "rule.base",
        "missing_component": False,
    },
    {
        "id": "group",
        "question": "Count by rated. Output fields: rated, n.",
        "raw_output": '{"collection":"movies","pipeline":[{"$group":{"_id":"$rated"}}]}',
        "error_type": "wrong_result",
        "error": "missing n accumulator",
        "implicated_component_id": None,
        "missing_component": True,
    },
]


def case_to_trace(case: dict, harness_id: str = "h0") -> Trace:
    tid = f"attr_{case['id']}"
    return Trace(
        trace_id=f"tr_{tid}",
        harness_id=harness_id,
        generation=0,
        task_id=tid,
        context_components=[
            ComponentRef(component_id="rule.base", version=1),
            ComponentRef(component_id="rule.types", version=1),
            ComponentRef(component_id="policy.cot", version=1),
            ComponentRef(component_id="policy.n_sample_docs", version=1),
        ],
        raw_output=case["raw_output"],
        error=case["error"],
        error_type=case["error_type"],
        score=0.0,
    )
