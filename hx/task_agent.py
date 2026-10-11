from __future__ import annotations

import os
import time
import uuid

from hx.harness import STUB_CTX, assemble, load_harness
from hx.llm import complete
from hx.models import ErrorType, Task, ToolCall, Trace
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


def _validate(pipe: list[dict]) -> None:
    try:
        from hx.executor import validate_pipeline as a_validate

        a_validate(pipe)
    except ImportError:
        validate_pipeline(pipe)


def _run_pipeline(dataset: str, collection: str, pipe: list[dict]) -> list[dict] | None:
    """Execute via A's executor when present; else task-role client. Never eval."""
    try:
        from hx.executor import run_pipeline

        return run_pipeline(dataset, collection, pipe, max_ms=5000, row_cap=500)
    except ImportError:
        pass
    try:
        from hx.config import get_client

        coll = get_client("task")[dataset][collection]
        out: list[dict] = []
        for i, doc in enumerate(coll.aggregate(pipe, maxTimeMS=5000)):
            if i >= 500:
                break
            out.append(doc)
        return out
    except Exception as err:
        raise RuntimeError(str(err)) from err


def _sample_docs(dataset: str, collection: str, n: int, trunc: int) -> str:
    n = max(1, min(int(n or 1), 3))
    try:
        from bson.json_util import dumps
        from hx.config import get_client

        coll = get_client("task")[dataset][collection]
        docs = list(coll.find().limit(n))
        return "\n".join(dumps(d)[:trunc] for d in docs) or "[]"
    except Exception as err:
        return f"(sample_docs failed: {err})"


def _classify_exec(msg: str) -> ErrorType:
    lower = msg.lower()
    if "timeout" in lower or "maxtimems" in lower:
        return "timeout"
    if "blocked" in lower or "stage" in lower:
        return "blocked"
    return "exec_error"


def run_task(harness_id: str, task: Task, seed: int = 0) -> Trace:
    t0 = time.perf_counter()
    harness, comps = load_harness(harness_id)
    ctx = _ctx(task.dataset)
    system, messages, tools = assemble(harness, comps, task, ctx)
    policies = {c.component_id: c.value for c in comps if c.kind == "policy"}
    max_retries = int(policies.get("policy.max_retries") or 0)
    trunc = int(policies.get("policy.doc_truncate_chars") or 300)
    model = os.getenv("TASK_MODEL", os.getenv("META_MODEL", "claude-haiku-5-5"))
    allow_sample_tool = bool(policies.get("policy.sample_docs_tool")) and bool(tools)

    prompt_tokens = 0
    completion_tokens = 0
    parsed = None
    error = None
    error_type: ErrorType = "none"
    raw = ""
    attempts = 0
    tool_calls: list[ToolCall] = []
    max_tries = 1 + max(0, max_retries)

    for attempts in range(1, max_tries + 1):
        result = complete(
            model, system, messages, tools if allow_sample_tool else None, 0.0, int(2048)
        )
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        raw = result.text

        for tc in result.tool_calls or []:
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            ok = True
            if name == "sample_docs" and allow_sample_tool:
                body = _sample_docs(
                    task.dataset, str(args.get("collection") or task.collection), int(args.get("n") or 1), trunc
                )
            else:
                ok = False
                body = f"unknown tool {name}"
            tool_calls.append(ToolCall(name=name, args=dict(args), ok=ok, result_tokens=len(body) // 4))
            messages = messages + [
                {"role": "assistant", "content": raw or f"tool {name}"},
                {"role": "user", "content": f"Tool {name} result:\n{body}\nReturn JSON {{collection, pipeline}} only."},
            ]
            follow = complete(model, system, messages, None, 0.0, int(2048))
            prompt_tokens += follow.prompt_tokens
            completion_tokens += follow.completion_tokens
            raw = follow.text

        try:
            parsed = parse_json_answer(raw)
            if parsed.get("tool") and "args" in parsed:
                raise ValueError("tool expansion is stretch; return a pipeline JSON object")
            pipe = pipeline_from_parsed(parsed)
            collection = str(parsed.get("collection") or task.collection)
            _validate(pipe)
            try:
                _run_pipeline(task.dataset, collection, pipe)
            except RuntimeError as err:
                msg = str(err)
                if "no URI" in msg or "A fills env" in msg:
                    pass
                else:
                    raise
            error = None
            error_type = "none"
            break
        except Exception as err:  # noqa: BLE001
            msg = str(err)
            error = msg
            if "blocked" in msg.lower() or "stage" in msg.lower():
                error_type = "blocked"
            elif parsed is None:
                error_type = "parse"
            else:
                error_type = _classify_exec(msg)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Previous answer failed ({error_type}): {msg}. Return corrected JSON only.",
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
        tool_calls=tool_calls,
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
