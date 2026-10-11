"""Parse / validate agent JSON. Never execute model output as code."""
from __future__ import annotations

import re
from typing import Any

from bson.json_util import loads

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
    parsed = loads(blob[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output is not a JSON object")
    return parsed


def pipeline_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = parsed.get("pipeline")
    if isinstance(raw, str):
        raw = loads(raw)
    if not isinstance(raw, list):
        raise ValueError("pipeline must be a list")
    for stage in raw:
        if not isinstance(stage, dict):
            raise ValueError("pipeline must be a list of dicts")
    return raw


def validate_pipeline(pipeline: list[dict[str, Any]]) -> None:
    if not isinstance(pipeline, list):
        raise ValueError("pipeline must be a list of dicts")
    if len(pipeline) > MAX_STAGES:
        raise ValueError(f"max {MAX_STAGES} stages")
    for stage in pipeline:
        if not isinstance(stage, dict):
            raise ValueError(f"invalid stage {stage!r}")
        keys = set(stage)
        blocked = keys & BLOCKED
        if blocked:
            raise ValueError(f"blocked stage {blocked}")
