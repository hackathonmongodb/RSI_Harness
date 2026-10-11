from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rsi_harness.db import db


def active() -> dict[str, Any]:
    doc = db().harness_versions.find_one({"status": "production"}, sort=[("version", -1)])
    if not doc:
        raise RuntimeError("No production harness. Run: python -m rsi_harness.seed")
    return doc


def list_versions(limit: int = 20) -> list[dict[str, Any]]:
    return list(db().harness_versions.find().sort("version", -1).limit(limit))


def insert_version(payload: dict[str, Any]) -> dict[str, Any]:
    last = db().harness_versions.find_one(sort=[("version", -1)])
    version = (last["version"] if last else 0) + 1
    doc = {
        **payload,
        "version": version,
        "created_at": datetime.now(timezone.utc),
    }
    db().harness_versions.insert_one(doc)
    return doc


def promote(version: int) -> None:
    db().harness_versions.update_many(
        {"status": "production"}, {"$set": {"status": "retired"}}
    )
    res = db().harness_versions.update_one(
        {"version": version}, {"$set": {"status": "production"}}
    )
    if res.matched_count == 0:
        raise RuntimeError(f"harness version {version} not found")


def record_eval(harness_version: int, task: str, score: float, **extra: Any) -> dict[str, Any]:
    doc = {
        "harness_version": harness_version,
        "task": task,
        "score": score,
        "latency_ms": extra.get("latency_ms", 0),
        "tokens_used": extra.get("tokens_used", 0),
        "guardrail_violations": extra.get("guardrail_violations", 0),
        "notes": extra.get("notes", ""),
        "created_at": datetime.now(timezone.utc),
    }
    db().eval_runs.insert_one(doc)
    avg = list(
        db().eval_runs.aggregate(
            [
                {"$match": {"harness_version": harness_version}},
                {"$group": {"_id": None, "avg": {"$avg": "$score"}}},
            ]
        )
    )
    if avg:
        db().harness_versions.update_one(
            {"version": harness_version}, {"$set": {"avg_score": avg[0]["avg"]}}
        )
    return doc
