"""Read-only pipeline validation and execution (task role). SPEC §7."""

from __future__ import annotations

from typing import Any

from hx.config import get_client

BLOCKED_STAGES = frozenset({"$out", "$merge", "$function", "$accumulator", "$where"})
MAX_STAGES = 12


def validate_pipeline(pipeline: list[dict]) -> None:
    """Raise ValueError if pipeline is unsafe or malformed."""
    if not isinstance(pipeline, list):
        raise ValueError("pipeline must be a list of dicts")
    if len(pipeline) > MAX_STAGES:
        raise ValueError(f"max {MAX_STAGES} stages")
    for stage in pipeline:
        if not isinstance(stage, dict) or not stage:
            raise ValueError("pipeline must be a list of non-empty dicts")
        keys = set(stage)
        blocked = keys & BLOCKED_STAGES
        if blocked:
            raise ValueError(f"blocked stage {sorted(blocked)}")
        # nested operator forms occasionally appear as value-level; scan shallowly
        for v in stage.values():
            if isinstance(v, dict):
                nested = set(v) & BLOCKED_STAGES
                if nested:
                    raise ValueError(f"blocked stage {sorted(nested)}")


def run_pipeline(
    dataset: str,
    collection: str,
    pipeline: list[dict],
    max_ms: int = 5000,
    row_cap: int = 500,
) -> list[dict]:
    """Run aggregation with the read-only task user. Caps time and rows."""
    validate_pipeline(pipeline)
    coll = get_client("task")[dataset][collection]
    out: list[dict[str, Any]] = []
    try:
        cursor = coll.aggregate(pipeline, maxTimeMS=int(max_ms))
        for i, doc in enumerate(cursor):
            if i >= int(row_cap):
                break
            out.append(doc)
    except Exception as err:  # noqa: BLE001 — surface to scorer/task_agent
        msg = str(err)
        lower = msg.lower()
        if "maxtimems" in lower or "exceeded time limit" in lower or "timeout" in lower:
            raise TimeoutError(msg) from err
        raise RuntimeError(msg) from err
    return out
