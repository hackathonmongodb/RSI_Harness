from __future__ import annotations

import os
import time
import uuid

from hx.harness import STUB_CTX, assemble, load_harness
from hx.llm import complete
from hx.models import ErrorType, Task, Trace
from hx.pipeline import parse_json_answer, pipeline_from_parsed, validate_pipeline


def _ctx(dataset: str) -> dict:
    try:
        from hx.schema_probe import get_dataset_context

        return get_dataset_context(dataset)
    except Exception:
        return STUB_CTX


def _score(task: Task, parsed: dict | None) -> tuple[float, ErrorType, str | None]:
    try:
        from hx.scorer import score

        return score(task, parsed)
    except Exception:
        if not parsed or "pipeline" not in parsed:
            return 0.0, "parse", "no pipeline"
        return 1.0, "none", None


def run_task(harness_id: str, task: Task, seed: int = 0) -> Trace:
    t0 = time.perf_counter()
    harness, comps = load_harness(harness_id)
    ctx = _ctx(task.dataset)
    system, messages, tools = assemble(harness, comps, task, ctx)
    policies = {c.component_id: c.value for c in comps if c.kind == "policy"}
    max_retries = int(policies.get("policy.max_retries") or 0)
    model = os.getenv("TASK_MODEL", "claude-haiku-5-5")

    prompt_tokens = 0
    completion_tokens = 0
    parsed = None
    error = None
    error_type: ErrorType = "none"
    raw = ""
    attempts = 0
    max_tries = 1 + max(0, max_retries)

    for attempts in range(1, max_tries + 1):
        result = complete(model, system, messages, tools, 0.0, int(2048))
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        raw = result.text
        try:
            parsed = parse_json_answer(raw)
            pipe = pipeline_from_parsed(parsed)
            try:
                from hx.executor import validate_pipeline as a_validate

                a_validate(pipe)
            except ImportError:
                validate_pipeline(pipe)
            error = None
            error_type = "none"
            break
        except Exception as err:  # noqa: BLE001
            msg = str(err)
            error = msg
            error_type = "blocked" if "blocked" in msg.lower() or "stage" in msg.lower() else "parse"
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Previous answer failed validation: {msg}. Return corrected JSON only.",
                },
            ]

    score, scored_err, err_detail = _score(task, parsed)
    if error is None:
        error_type = scored_err
        error = err_detail
    elif score == 0.0 and error_type == "none":
        error_type = "parse"

    return Trace(
        trace_id=uuid.uuid4().hex,
        harness_id=harness.harness_id,
        generation=harness.generation,
        task_id=task.task_id,
        seed=seed,
        context_components=list(harness.components),
        raw_output=raw,
        parsed=parsed,
        attempts=attempts,
        error=error,
        error_type=error_type,
        score=score,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
