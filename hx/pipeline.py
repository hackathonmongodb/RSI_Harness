from __future__ import annotations

import json
import re
from typing import Any


WRITE_KEYS = {"$out", "$merge"}


def parse_json_answer(text: str) -> dict[str, Any]:
    blob = text.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(blob[start : end + 1])


def pipeline_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    op = (parsed.get("operation") or "aggregate").lower()
    if op == "aggregate":
        raw = parsed.get("pipeline")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            raise ValueError("aggregate requires a list pipeline")
        return raw
    filt = parsed.get("filter") or {}
    stages: list[dict[str, Any]] = [{"$match": filt}] if filt else []
    if parsed.get("sort"):
        stages.append({"$sort": parsed["sort"]})
    if parsed.get("limit") is not None:
        stages.append({"$limit": int(parsed["limit"])})
    if parsed.get("projection"):
        stages.append({"$project": parsed["projection"]})
    return stages or [{"$match": {}}]


def validate_pipeline(pipeline: list[dict[str, Any]]) -> None:
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("pipeline must be a non-empty list")
    for stage in pipeline:
        if not isinstance(stage, dict) or not stage:
            raise ValueError(f"invalid stage {stage!r}")
        if set(stage) & WRITE_KEYS:
            raise ValueError("write stages are not allowed")
