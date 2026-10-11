from __future__ import annotations

from typing import Any

from hx.contracts import load_executor, load_scorer, structural_score
from hx.harness import assemble
from hx.llm import complete
from hx.models import Harness, Task, Trace
from hx.pipeline import parse_json_answer, pipeline_from_parsed, validate_pipeline


def run_task(
    harness: Harness,
    task: Task,
    seed: int = 0,
    model: str | None = None,
) -> Trace:
    """Build prompt, call model at temperature 0, parse JSON, retry, score, return Trace."""
    assembled = assemble(harness, task)
    messages = list(assembled.messages)
    total_in = 0
    total_out = 0
    parsed: dict[str, Any] | None = None
    pipeline = None
    error: str | None = None
    raw = ""
    attempts = 0
    max_tries = 1 + max(0, harness.policy.max_retries)

    for attempts in range(max_tries):
        result = complete(
            model,
            assembled.system,
            messages,
            assembled.tools,
            0.0,
            harness.policy.max_tokens,
        )
        total_in += result.input_tokens
        total_out += result.output_tokens
        raw = result.text
        try:
            parsed = parse_json_answer(raw)
            pipeline = pipeline_from_parsed(parsed)
            validate_pipeline(pipeline)
            error = None
            break
        except Exception as err:  # noqa: BLE001
            error = str(err)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Your previous answer failed validation: {error}. Return corrected JSON only.",
                },
            ]

    retries = max(0, attempts if error is None else attempts)
    exec_result = None
    executor = load_executor()
    if pipeline is not None and executor is not None:
        try:
            exec_result = executor.execute(task, pipeline)
        except Exception as err:  # noqa: BLE001
            error = f"executor: {err}"

    scorer = load_scorer()
    if scorer is not None:
        score = scorer.score(task, parsed, exec_result)
    else:
        score = structural_score(parsed, error)

    return Trace(
        harness_id=harness.harness_id,
        harness_version=harness.version,
        task_id=task.task_id,
        seed=seed,
        split=task.split,
        natural_language=task.nl,
        context_components=assembled.context_components,
        raw_response=raw,
        parsed=parsed,
        pipeline=pipeline,
        retries=retries,
        score=score,
        input_tokens=total_in,
        output_tokens=total_out,
        error=error,
        executor_result=exec_result,
    )
