from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from hx.contracts import load_tasks
from hx.fallback_tasks import FALLBACK_TASKS
from hx.models import GenerationReport, Harness, Task, Trace
from hx.task_agent import run_task

_CACHE: dict[tuple, GenerationReport] = {}


def _task_set(task_ids: list[str] | None = None) -> list[Task]:
    tasks = load_tasks() or list(FALLBACK_TASKS)
    if task_ids:
        want = set(task_ids)
        tasks = [t for t in tasks if t.task_id in want]
    return tasks


def _cache_key(harness_id: str, seed: int, tasks: list[Task]) -> tuple:
    ids = tuple(sorted(t.task_id for t in tasks))
    return (harness_id, seed, ids)


def _persist_trace(trace: Trace) -> None:
    try:
        from hx.config import get_client, mongodb_db
    except Exception:
        return
    try:
        db = get_client()[mongodb_db()]
    except Exception:
        return
    doc = trace.model_dump()
    if trace.split == "train":
        db.get_collection("hx_evidence").update_one(
            {
                "harness_id": trace.harness_id,
                "task_id": trace.task_id,
                "seed": trace.seed,
            },
            {"$set": doc},
            upsert=True,
        )
    else:
        db.get_collection("hx_eval.heldout_results").update_one(
            {
                "harness_id": trace.harness_id,
                "task_id": trace.task_id,
                "seed": trace.seed,
            },
            {
                "$set": {
                    "harness_id": trace.harness_id,
                    "task_id": trace.task_id,
                    "seed": trace.seed,
                    "score": trace.score,
                    "input_tokens": trace.input_tokens,
                    "output_tokens": trace.output_tokens,
                    "error": trace.error,
                }
            },
            upsert=True,
        )


def _update_harness_scores(harness: Harness, report: GenerationReport) -> None:
    def _mean(vals: list[float | None]) -> float | None:
        nums = [v for v in vals if v is not None]
        return sum(nums) / len(nums) if nums else None

    payload = {
        "train_mean": _mean(list(report.train_scores.values())),
        "heldout_mean": _mean(list(report.heldout_scores.values())),
        "n_train": len(report.train_scores),
        "n_heldout": len(report.heldout_scores),
        "seed": report.seed,
    }
    harness.scores = payload
    try:
        from hx.config import get_client, mongodb_db

        db = get_client()[mongodb_db()]
        db.harnesses.update_one(
            {"harness_id": harness.harness_id, "version": harness.version},
            {"$set": {"scores": payload}},
        )
    except Exception:
        return


def run_generation(
    harness: Harness,
    seed: int = 0,
    task_ids: list[str] | None = None,
    persist: bool = False,
    max_workers: int | None = None,
) -> GenerationReport:
    tasks = _task_set(task_ids)
    key = _cache_key(harness.harness_id, seed, tasks)
    if key in _CACHE:
        cached = _CACHE[key].model_copy()
        cached.cached = True
        return cached

    workers = max_workers or int(os.getenv("HX_CONCURRENCY", "4"))
    t0 = time.perf_counter()
    traces: list[Trace] = []
    errors: list[BaseException] = []

    def _one(task: Task) -> Trace:
        return run_task(harness, task, seed=seed)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                traces.append(fut.result())
            except Exception as err:  # noqa: BLE001
                if "429" in str(err) or "rate" in str(err).lower():
                    errors.append(err)
                else:
                    errors.append(err)

    if errors and traces == []:
        raise errors[0]

    traces.sort(key=lambda tr: tr.task_id)
    train = {tr.task_id: tr.score for tr in traces if tr.split == "train"}
    held = {tr.task_id: tr.score for tr in traces if tr.split == "heldout"}
    tokens = {tr.task_id: tr.input_tokens + tr.output_tokens for tr in traces}
    report = GenerationReport(
        harness_id=harness.harness_id,
        seed=seed,
        train_scores=train,
        heldout_scores=held,
        tokens_per_task=tokens,
        traces=traces,
        elapsed_s=time.perf_counter() - t0,
        cached=False,
    )
    if persist:
        for tr in traces:
            _persist_trace(tr)
        _update_harness_scores(harness, report)
    else:
        harness.scores = {
            "train_mean": _mean_safe(list(train.values())),
            "heldout_mean": _mean_safe(list(held.values())),
            "seed": seed,
        }
    _CACHE[key] = report
    return report


def _mean_safe(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def clear_cache() -> None:
    _CACHE.clear()
