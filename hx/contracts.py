"""Interfaces A should implement. B imports these names if hx.executor / hx.scorer / hx.tasks exist."""

from __future__ import annotations

from typing import Any, Protocol

from hx.models import Task


class Executor(Protocol):
    def execute(self, task: Task, pipeline: list[dict[str, Any]]) -> Any: ...


class Scorer(Protocol):
    def score(self, task: Task, parsed: dict[str, Any] | None, exec_result: Any) -> float: ...


def load_executor() -> Executor | None:
    try:
        from hx.executor import executor  # owned by A
    except ImportError:
        return None
    return executor


def load_scorer() -> Scorer | None:
    try:
        from hx.scorer import scorer
    except ImportError:
        return None
    return scorer


def load_tasks() -> list[Task] | None:
    try:
        from hx.tasks import TASKS
    except ImportError:
        return None
    return list(TASKS)


def structural_score(parsed: dict | None, error: str | None) -> float:
    if error or not parsed:
        return 0.0
    if parsed.get("pipeline") or parsed.get("filter") is not None or parsed.get("operation"):
        return 1.0
    return 0.0
