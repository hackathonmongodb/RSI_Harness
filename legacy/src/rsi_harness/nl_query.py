"""Natural Language → MongoDB, following MongoDB's official prompting guide.

https://www.mongodb.com/docs/manual/natural-language-to-mongodb/
Official skill: skills/mongodb-natural-language-querying/SKILL.md
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import json5
from bson import ObjectId
from pymongo.collection import Collection

from rsi_harness.db import db
from rsi_harness.introspect import database_prompt
from rsi_harness.llm import complete, parse_json_object

# Official base prompt (MongoDB docs) plus a JSON envelope so we can execute safely.
BASE_PROMPT = """You are an expert data analyst experienced at using MongoDB.
Your job is to take information about a MongoDB database plus a natural language query and generate a MongoDB shell (mongosh) query to execute to retrieve the information needed to answer the natural language query.

Format the mongosh query in the following structure:

`db.<collection name>.find({/* query */})` or `db.<collection name>.aggregate([/* query */])`

Some general query-authoring tips:

1. Ensure proper use of MongoDB operators ($eq, $gt, $lt, etc.) and data types (ObjectId, ISODate).
2. For complex queries, use aggregation pipeline with proper stages ($match, $group, $lookup, etc.).
3. Consider performance by utilizing available indexes, avoiding $where and full collection scans, and using covered queries where possible.
4. Include sorting (.sort()) and limiting (.limit()) when appropriate for result set management.
5. Handle null values and existence checks explicitly with $exists and $type operators to differentiate between missing fields, null values, and empty arrays.
6. Do not include `null` in results objects in aggregation, e.g. do not include _id: null.
7. For date operations, NEVER use an empty new date object (e.g. `new Date()`). ALWAYS specify the date, such as `new Date("2024-10-24")`. Use the provided 'Latest Date' field to inform dates in queries.
8. For Decimal128 operations, prefer range queries over exact equality.
9. When querying arrays, use appropriate operators like $elemMatch for complex matching, $all to match multiple elements, or $size for array length checks.

Think step by step about the code in the answer before providing it. In your thoughts, consider:
1. Which collections are relevant to the query.
2. Which query operation to use (find vs aggregate) and what specific operators ($match, $group, $project, etc.) are needed.
3. What fields are relevant to the query.
4. Which indexes you can use to improve performance.
5. What specific transformations or projections are required.
6. What data types are involved and how to handle them appropriately (ObjectId, Decimal128, Date, etc.).
7. What edge cases to consider (empty results, null values, missing fields).
8. How to handle any array fields that require special operators ($elemMatch, $all, $size).
9. Any other relevant considerations.

READ-ONLY: never generate insert/update/delete/drop/$out/$merge/$lookup that writes.

Return ONLY a JSON object (no markdown) with this shape:
{
  "thoughts": "brief chain of thought",
  "mongosh": "db.<collection>.find(...) or db.<collection>.aggregate([...])",
  "collection": "collection_name",
  "operation": "find" or "aggregate",
  "filter": {},
  "projection": {},
  "sort": {},
  "skip": 0,
  "limit": 20,
  "pipeline": []
}
Use strict JSON (double-quoted keys). filter/projection/sort/pipeline must be executable Extended JSON.
Prefer find over aggregate unless grouping, $lookup, or multi-stage transforms are required.
"""

FEW_SHOT = """
**Example 1**

Natural language query: Show me the 5 latest production harness versions

mongosh:
db.harness_versions.find({ status: 'production' }).sort({ created_at: -1 }).limit(5)

**Example 2**

Natural language query: Average eval score by harness version, lowest first

mongosh:
db.eval_runs.aggregate([
  { $group: { _id: '$harness_version', avg_score: { $avg: '$score' }, n: { $sum: 1 } } },
  { $sort: { avg_score: 1 } }
])

**Example 3**

Natural language query: List memories tagged long-horizon from the last 7 days

mongosh:
db.memories.find({ tags: 'long-horizon' }).sort({ created_at: -1 }).limit(20)
"""

def _sanitize(spec: dict[str, Any]) -> dict[str, Any]:
    pipeline = spec.get("pipeline") or []
    for stage in pipeline:
        if isinstance(stage, dict) and ({k.lower() for k in stage} & {"$out", "$merge"}):
            raise ValueError("Write stages ($out/$merge) are not allowed")
    mongosh = str(spec.get("mongosh") or "").lower()
    banned = ("insert", "updateone", "updatemany", "deleteone", "deletemany", "drop(", "$out", "$merge")
    if any(tok in mongosh for tok in banned):
        raise ValueError("Write operations are not allowed in NL queries")
    return spec


def generate(natural_language: str, collections: list[str] | None = None) -> dict[str, Any]:
    user = f"""Generate MongoDB Shell (mongosh) queries for the following database and natural language query:

{database_prompt(collections)}

{FEW_SHOT}

Natural language query: {natural_language}
"""
    raw = complete(BASE_PROMPT, user)
    spec = parse_json_object(raw)
    spec["natural_language"] = natural_language
    return _sanitize(spec)


def _oidify(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"$oid"}:
            return ObjectId(obj["$oid"])
        if set(obj) == {"$date"}:
            v = obj["$date"]
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return {k: _oidify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_oidify(x) for x in obj]
    return obj


def execute_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    col_name = spec["collection"]
    col: Collection = db()[col_name]
    op = spec.get("operation") or "find"
    if op == "aggregate":
        pipeline = _oidify(spec.get("pipeline") or [])
        return list(col.aggregate(pipeline))
    filt = _oidify(spec.get("filter") or {})
    proj = spec.get("projection") or None
    cursor = col.find(filt, proj)
    sort = spec.get("sort")
    if sort:
        cursor = cursor.sort(list(sort.items()) if isinstance(sort, dict) else sort)
    skip = int(spec.get("skip") or 0)
    if skip:
        cursor = cursor.skip(skip)
    limit = int(spec.get("limit") or 50)
    cursor = cursor.limit(min(limit, 100))
    return list(cursor)


def run(natural_language: str, collections: list[str] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = generate(natural_language, collections)
    docs = execute_spec(spec)
    ms = int((time.perf_counter() - t0) * 1000)
    from bson.json_util import dumps as bson_dumps

    serializable = json5.loads(bson_dumps(docs))
    trace = {
        "natural_language": natural_language,
        "collection": spec.get("collection"),
        "mongosh": spec.get("mongosh"),
        "operation": spec.get("operation"),
        "executable": {
            "filter": spec.get("filter"),
            "projection": spec.get("projection"),
            "sort": spec.get("sort"),
            "limit": spec.get("limit"),
            "pipeline": spec.get("pipeline"),
        },
        "thoughts": spec.get("thoughts"),
        "result_count": len(docs),
        "latency_ms": ms,
        "created_at": datetime.now(timezone.utc),
    }
    db().query_traces.insert_one(trace)
    return {
        "natural_language": natural_language,
        "mongosh": spec.get("mongosh"),
        "thoughts": spec.get("thoughts"),
        "operation": spec.get("operation"),
        "collection": spec.get("collection"),
        "result_count": len(docs),
        "latency_ms": ms,
        "results": serializable[:40],
    }
