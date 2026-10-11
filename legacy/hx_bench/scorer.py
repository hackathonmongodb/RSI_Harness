"""Normalize + score candidate pipelines against gold. SPEC §3 / §7."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from bson import Decimal128, Int64, ObjectId
from bson.binary import Binary

from hx.config import get_client
from hx.executor import run_pipeline, validate_pipeline
from hx.models import ErrorType, Split, Task


def _to_plain(value: Any) -> Any:
    if isinstance(value, Decimal128):
        return round(float(value.to_decimal()), 4)
    if isinstance(value, Int64):
        return round(float(int(value)), 4)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return round(float(value), 4)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat() + (
            "Z" if value.tzinfo is None else ""
        )
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Binary):
        return value.hex()
    if isinstance(value, dict):
        return {k: _to_plain(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def normalize(docs: list[dict]) -> list[str]:
    """Canonical JSON strings per doc (sorted keys, typed numbers rounded)."""
    out: list[str] = []
    for doc in docs:
        plain = _to_plain(doc)
        if not isinstance(plain, dict):
            plain = {"_value": plain}
        out.append(json.dumps(plain, sort_keys=True, separators=(",", ":")))
    return out


def _equal(cand: list[dict], gold: list[dict], ordered: bool) -> bool:
    a = normalize(cand)
    b = normalize(gold)
    if ordered:
        return a == b
    return sorted(a) == sorted(b)


def _load_gold_pipeline(task_id: str) -> list[dict] | None:
    doc = get_client("admin")["hx_eval"]["gold"].find_one({"task_id": task_id})
    if not doc:
        return None
    pipe = doc.get("pipeline")
    return pipe if isinstance(pipe, list) else None


def score(task: Task, parsed: dict | None) -> tuple[float, ErrorType, str | None]:
    """Run candidate vs gold; return (0|1, error_type, detail)."""
    if not parsed or not isinstance(parsed, dict):
        return 0.0, "parse", "no parsed output"
    pipe = parsed.get("pipeline")
    if isinstance(pipe, str):
        try:
            from bson.json_util import loads

            pipe = loads(pipe)
        except Exception as err:  # noqa: BLE001
            return 0.0, "parse", f"pipeline not valid EJSON: {err}"
    if not isinstance(pipe, list):
        return 0.0, "parse", "pipeline must be a list"
    try:
        validate_pipeline(pipe)
    except ValueError as err:
        return 0.0, "blocked", str(err)

    collection = str(parsed.get("collection") or task.collection)
    try:
        cand = run_pipeline(task.dataset, collection, pipe, max_ms=5000, row_cap=500)
    except TimeoutError as err:
        return 0.0, "timeout", str(err)
    except Exception as err:  # noqa: BLE001
        return 0.0, "exec_error", str(err)

    gold_pipe = _load_gold_pipeline(task.task_id)
    if gold_pipe is None:
        return 0.0, "exec_error", f"missing gold for {task.task_id}"
    try:
        gold_rows = run_pipeline(
            task.dataset, task.collection, gold_pipe, max_ms=5000, row_cap=500
        )
    except Exception as err:  # noqa: BLE001
        return 0.0, "exec_error", f"gold exec failed: {err}"

    if _equal(cand, gold_rows, ordered=bool(task.ordered)):
        return 1.0, "none", None
    return (
        0.0,
        "wrong_result",
        f"mismatch (cand={len(cand)} gold={len(gold_rows)} ordered={task.ordered})",
    )


def get_noise_floor(dataset: str) -> float:
    """Promotion margin = 2 * std from hx_eval.noise (SPEC §3 / C request)."""
    doc = get_client("admin")["hx_eval"]["noise"].find_one({"dataset": dataset})
    if not doc:
        return 0.05  # default until noise_floor.py runs
    if "margin" in doc:
        return float(doc["margin"])
    std = float(doc.get("std") or 0.0)
    return 2.0 * std


def load_tasks(dataset: str, split: Split) -> list[Task]:
    """Load tasks for a dataset/split (no gold). Re-exported for B's import path."""
    from hx.schema_probe import load_tasks as _load

    return _load(dataset, split)
