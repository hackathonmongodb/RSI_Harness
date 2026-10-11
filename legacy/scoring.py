"""LLM-as-judge scorer (Stream A). Separate from hx.scorer (gold pipeline equality)."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm import complete

log = logging.getLogger("scoring")

JUDGE_SYSTEM = """You are a strict grader for a MongoDB / data harness.
Score the agent's answer against the task (and reference if given).
Return ONLY a JSON object with keys:
  score: number from 0 to 10
  pass: boolean (true if score >= 7)
  reasoning: short string
"""


def _parse_verdict(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in judge output: {text[:240]}")
    data = json.loads(raw[start : end + 1])
    if "score" not in data or "pass" not in data or "reasoning" not in data:
        raise ValueError(f"missing keys in judge JSON: {data}")
    score = data["score"]
    if score is not None:
        score = float(score)
        if score < 0 or score > 10:
            raise ValueError(f"score out of range: {score}")
    data["score"] = score
    data["pass"] = bool(data["pass"])
    data["reasoning"] = str(data["reasoning"])
    return {
        "score": data["score"],
        "pass": data["pass"],
        "reasoning": data["reasoning"],
    }


def score(
    task: dict[str, Any],
    agent_output: str,
    reference: str | None = None,
) -> dict[str, Any]:
    """Judge agent_output for task. Returns {score, pass, reasoning} or score=null on failure."""
    ref = reference if reference is not None else task.get("reference")
    user = {
        "task_id": task.get("id"),
        "prompt": task.get("prompt") or task.get("question"),
        "reference": ref,
        "agent_output": agent_output,
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(user)},
    ]
    try:
        text = complete(messages, temperature=0, json_mode=True)
        return _parse_verdict(text)
    except Exception as first_err:  # noqa: BLE001
        log.warning("judge parse/call failed once: %s — retrying", first_err)
        try:
            text = complete(
                messages
                + [
                    {
                        "role": "user",
                        "content": "Retry: return valid JSON only with score, pass, reasoning.",
                    }
                ],
                temperature=0,
                json_mode=True,
            )
            return _parse_verdict(text)
        except Exception as err:  # noqa: BLE001
            return {
                "score": None,
                "pass": False,
                "reasoning": f"judge_error: {err}",
                "error": str(err),
            }
