"""Dataset context + task loading. SPEC §7."""

from __future__ import annotations

from typing import Any

from bson.json_util import RELAXED_JSON_OPTIONS, dumps

from hx.config import get_client
from hx.models import Split, Task

# Prefer these collections when present (sample_mflix); others discovered at runtime.
_PREFERRED = {
    "sample_mflix": ["movies", "comments", "theaters", "users"],
    "sample_analytics": ["customers", "accounts", "transactions"],
}


def _infer_ts_type(value: Any) -> str:
    from datetime import datetime

    from bson import Decimal128, Int64, ObjectId

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, ObjectId):
        return "ObjectId"
    if isinstance(value, Decimal128):
        return "Decimal128"
    if isinstance(value, Int64):
        return "number /* Int64 */"
    if isinstance(value, int):
        return "number /* int */"
    if isinstance(value, float):
        return "number /* float */"
    if isinstance(value, datetime):
        return "Date"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        if not value:
            return "any[]"
        return f"{_infer_ts_type(value[0])}[]"
    if isinstance(value, dict):
        return "object"
    return "any"


def _ts_schema_from_docs(docs: list[dict]) -> str:
    fields: dict[str, str] = {}
    for doc in docs:
        for k, v in doc.items():
            if k not in fields:
                fields[k] = _infer_ts_type(v)
    lines = ["{"]
    for k in sorted(fields):
        lines.append(f"  {k}: {fields[k]}; // field {k}")
    lines.append("}")
    return "\n".join(lines)


def _index_descriptions(coll) -> list[str]:
    out: list[str] = []
    try:
        for idx in coll.list_indexes():
            keys = idx.get("key") or {}
            key_s = ", ".join(f"{k}:{v}" for k, v in keys.items())
            name = idx.get("name", "?")
            out.append(f"{name} ({key_s})")
    except Exception:  # noqa: BLE001
        pass
    return out


def get_dataset_context(dataset: str) -> dict:
    """Build/cache context for assemble(): description, collections, schemas, samples, indexes.

    Schema is generated once from sample docs and cached in hx_harness.dataset_context.
    Uses a heuristic TypeScript annotation (no LLM required) so B/C are unblocked;
    regenerate only by deleting the cached doc.
    """
    admin = get_client("admin")
    cache = admin["hx_harness"]["dataset_context"]
    cached = cache.find_one({"dataset": dataset})
    if cached and cached.get("context"):
        return cached["context"]

    task = get_client("task")
    db = task[dataset]
    names = list(db.list_collection_names())
    prefer = _PREFERRED.get(dataset, [])
    ordered = [n for n in prefer if n in names] + [
        n for n in sorted(names) if n not in prefer and not n.startswith("system.")
    ]

    collections: dict[str, dict] = {}
    for name in ordered:
        coll = db[name]
        samples = list(coll.find().limit(3))
        ejson_samples = [dumps(d, json_options=RELAXED_JSON_OPTIONS) for d in samples]
        collections[name] = {
            "description": f"Collection `{name}` in `{dataset}`.",
            "ts_schema": _ts_schema_from_docs(samples) if samples else "{}",
            "sample_docs": ejson_samples,
            "indexes": _index_descriptions(coll),
        }

    context = {
        "description": f"MongoDB sample dataset `{dataset}`.",
        "collections": collections,
    }
    cache.update_one(
        {"dataset": dataset},
        {"$set": {"dataset": dataset, "context": context}},
        upsert=True,
    )
    return context


def load_tasks(dataset: str, split: Split) -> list[Task]:
    """Load tasks for dataset/split from hx_eval.tasks (admin)."""
    cur = (
        get_client("admin")["hx_eval"]["tasks"]
        .find({"dataset": dataset, "split": split})
        .sort("task_id", 1)
    )
    return [Task.model_validate(doc) for doc in cur]
