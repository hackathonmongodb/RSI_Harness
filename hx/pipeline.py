from __future__ import annotations

import json
import re
from typing import Any

BLOCKED = {"$out", "$merge", "$function", "$accumulator", "$where"}
MAX_STAGES = 12


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
    raw = parsed.get("pipeline")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("pipeline must be a list")
    return raw


def validate_pipeline(pipeline: list[dict[str, Any]]) -> None:
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("pipeline must be a non-empty list")
    if len(pipeline) > MAX_STAGES:
        raise ValueError(f"max {MAX_STAGES} stages")
    for stage in pipeline:
        if not isinstance(stage, dict) or not stage:
            raise ValueError(f"invalid stage {stage!r}")
        if set(stage) & BLOCKED:
            raise ValueError(f"blocked stage {set(stage) & BLOCKED}")
