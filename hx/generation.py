from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from hx.fallback_tasks import FALLBACK_TASKS
from hx.models import GenerationResult, Task, Trace
from hx.task_agent import run_task

_CACHE: dict[tuple, GenerationResult] = {}


def _load_tasks(dataset: str, task_ids: list[str] | None) -> list[Task]:
    tasks: list[Task] = []
    try:
        from hx.scorer import load_tasks  # type: ignore
    except Exception:
        load_tasks = None
    try:
        from hx.tasks import TASKS

        tasks = [t for t in TASKS if t.dataset == dataset]
    except Exception:
        if load_tasks:
            try:
                tasks = list(load_tasks(dataset, "train")) + list(load_tasks(dataset, "heldout"))
            except Exception:
                tasks = []
        if not tasks:
            tasks = [t for t in FALLBACK_TASKS if t.dataset == dataset]
    if task_ids:
        want = set(task_ids)
        tasks = [t for t in tasks if t.task_id in want]
    return tasks


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _persist(trace: Trace, task: Task) -> None:
    try:
        from hx.config import get_client
    except Exception:
        return
    try:
        if task.split == "train":
            db = get_client("admin")["hx_evidence"]
            db.traces.update_one({"trace_id": trace.trace_id}, {"$set": trace.model_dump()}, upsert=True)
        else:
            db = get_client("admin")["hx_eval"]
            db.heldout_results.update_one(
                {"harness_id": trace.harness_id, "task_id": trace.task_id, "seed": trace.seed},
                {
                    "$set": {
                        "harness_id": trace.harness_id,
                        "task_id": trace.task_id,
                        "seed": trace.seed,
                        "score": trace.score,
                        "total_tokens": trace.prompt_tokens + trace.completion_tokens,
                    }
                },
                upsert=True,
            )
    except Exception:
        return


def _update_harness_doc(harness_id: str, result: GenerationResult) -> None:
    try:
        from hx.config import get_client
        from hx.harness import _HARNESSES

        h = _HARNESSES.get(harness_id)
        if h:
            h.train_score = result.train_score
            h.heldout_score = result.heldout_score
            h.tokens_per_task = result.tokens_per_task
        db = get_client("meta")["hx_harness"]
        db.harnesses.update_one(
            {"harness_id": harness_id},
            {
                "$set": {
                    "train_score": result.train_score,
                    "heldout_score": result.heldout_score,
                    "tokens_per_task": result.tokens_per_task,
                }
            },
        )
    except Exception:
        return


def run_generation(
    harness_id: str,
    dataset: str,
    seeds: list[int] | None = None,
    parallel: int | None = None,
    task_ids: list[str] | None = None,
) -> GenerationResult:
    seeds = seeds or [0]
    tasks = _load_tasks(dataset, task_ids)
    key = (harness_id, tuple(seeds), tuple(sorted(t.task_id for t in tasks)))
    if key in _CACHE:
        return _CACHE[key]

    workers = parallel or int(os.getenv("HX_CONCURRENCY", "8"))
    traces: list[tuple[Task, Trace]] = []

    def _one(pair: tuple[Task, int]) -> tuple[Task, Trace]:
        task, seed = pair
        return task, run_task(harness_id, task, seed=seed)

    jobs = [(t, s) for s in seeds for t in tasks]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_one, j) for j in jobs]
        for fut in as_completed(futs):
            traces.append(fut.result())

    train_outcomes: dict[str, float] = {}
    heldout_outcomes: dict[str, float] = {}
    failed: list[str] = []
    passed: list[str] = []
    tok: list[int] = []
    for task, tr in traces:
        tok.append(tr.prompt_tokens + tr.completion_tokens)
        _persist(tr, task)
        if task.split == "train":
            train_outcomes[tr.task_id] = tr.score
            if tr.score >= 1.0:
                passed.append(tr.trace_id)
            else:
                failed.append(tr.trace_id)
        else:
            heldout_outcomes[tr.task_id] = tr.score

    result = GenerationResult(
        harness_id=harness_id,
        train_score=_mean(list(train_outcomes.values())),
        heldout_score=_mean(list(heldout_outcomes.values())),
        tokens_per_task=_mean([float(x) for x in tok]),
        train_outcomes=train_outcomes,
        heldout_outcomes=heldout_outcomes,
        failed_train_trace_ids=failed,
        passed_train_trace_ids=passed,
    )
    _update_harness_doc(harness_id, result)
    _CACHE[key] = result
    return result


def clear_cache() -> None:
    _CACHE.clear()
