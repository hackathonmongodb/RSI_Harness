"""Layer-1 evidence. Public signature matches SPEC.md section 7."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _report_docs(harness_id: str) -> list[dict[str, Any]]:
    try:
        from tests.mocks import STORE

        return [r for r in STORE.reports if r.get("harness_id") == harness_id]
    except ImportError:
        return []


def build_overview(harness_id: str) -> dict[str, Any]:
    """Counts by implicated_component_id, kind, error_type, missing_component.

    Tags omitted until Atlas/admin can join hx_eval.tasks (no Trace.tags in models).
    """
    docs = _report_docs(harness_id)
    # Pipeline we would run on Atlas (kept here so tests can inspect intent):
    _facet_pipeline = [
        {"$match": {"harness_id": harness_id}},
        {
            "$facet": {
                "by_component": [
                    {"$group": {"_id": "$implicated_component_id", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                ],
                "by_kind": [{"$group": {"_id": "$implicated_kind", "n": {"$sum": 1}}}],
                "by_error_type": [{"$group": {"_id": "$error_type", "n": {"$sum": 1}}}],
                "missing_component": [
                    {"$match": {"missing_component": True}},
                    {"$count": "n"},
                ],
            }
        },
    ]
    by_component = Counter(d.get("implicated_component_id") for d in docs)
    by_kind = Counter(d.get("implicated_kind") for d in docs)
    by_error = Counter(d.get("error_type") for d in docs)
    missing = sum(1 for d in docs if d.get("missing_component"))
    overview = {
        "harness_id": harness_id,
        "n_reports": len(docs),
        "by_component": [{"id": k, "n": v} for k, v in by_component.most_common()],
        "by_kind": [{"id": k, "n": v} for k, v in by_kind.most_common()],
        "by_error_type": [{"id": k, "n": v} for k, v in by_error.most_common()],
        "missing_component_count": missing,
        "top_failing_tags": [],
        "_facet_pipeline": _facet_pipeline,
    }
    return overview
