"""Agent-driven recursive self-improve on sample_mflix.

Not a UI button — called by the ask agent when accuracy is below target.
Stops at target score, plateau, or max generations (see hx.loop.evolve).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from hx.llm import groq_api_key


def _auto_enabled() -> bool:
    return os.getenv("RSI_AUTO_EVOLVE", "1").strip().lower() not in {"0", "false", "no"}


def _target() -> float:
    try:
        return float(os.getenv("RSI_TARGET_SCORE", "0.85"))
    except ValueError:
        return 0.85


def current_train_score(dataset: str = "sample_mflix") -> float | None:
    """Best-effort current train score from harness store / mocks."""
    if os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}:
        try:
            from tests.mocks import STORE

            best = None
            for h in STORE.harnesses.values():
                if h.train_score is None:
                    continue
                if best is None or float(h.train_score) > best:
                    best = float(h.train_score)
            return best
        except Exception:
            return None
    try:
        from hx.config import get_client

        doc = get_client("meta")["hx_harness"]["harnesses"].find_one(
            {"dataset": dataset, "status": {"$in": ["promoted", "static", "production"]}},
            sort=[("generation", -1)],
        )
        if doc and doc.get("train_score") is not None:
            return float(doc["train_score"])
    except Exception:
        pass
    return None


def should_evolve(dataset: str = "sample_mflix") -> tuple[bool, str]:
    if not _auto_enabled():
        return False, "RSI_AUTO_EVOLVE disabled"
    if not groq_api_key():
        return False, "no GROQ_API_KEY — paste Groq key into .env"
    score = current_train_score(dataset)
    goal = _target()
    if score is not None and score >= goal:
        return False, f"already at target train={score:.3f} >= {goal:.3f}"
    return True, f"train={score} < target={goal}" if score is not None else f"below target={goal}"


def run_mflix_rsi(
    generations: int | None = None,
    *,
    target_score: float | None = None,
    plateau_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run agent evolve on sample_mflix with Llama + stop conditions."""
    ok, why = should_evolve("sample_mflix")
    if not force and not ok:
        return {
            "dataset": "sample_mflix",
            "skipped": True,
            "reason": why,
            "improved": False,
        }

    # Prefer live Llama; only mock the datastore loop when explicitly requested
    use_mocks = os.getenv("HX_USE_MOCKS", "").strip().lower() in {"1", "true"}
    if use_mocks:
        return _run_mocked(generations, target_score, plateau_limit)

    os.environ.setdefault("LLM_FAKE", "0")
    logging.basicConfig(level=logging.INFO)
    from hx.loop import evolve
    from hx.harness import load_harness

    stop: dict[str, Any] = {}
    start_id = os.getenv("RSI_START_HARNESS", "h0")
    try:
        start_h, _ = load_harness(start_id)
        start_train = start_h.train_score
    except Exception:
        start_train = None

    final_id = evolve(
        "sample_mflix",
        start_id,
        generations,
        target_score=target_score,
        plateau_limit=plateau_limit,
        stop_info=stop,
    )
    final, comps = load_harness(final_id)
    return {
        "dataset": "sample_mflix",
        "skipped": False,
        "mode": "groq llama agent RSI (live)",
        "provider": "groq",
        "start": {"harness_id": start_id, "train_score": start_train},
        "final": {
            "harness_id": final.harness_id,
            "generation": final.generation,
            "train_score": final.train_score,
            "heldout_score": final.heldout_score,
            "tokens_per_task": final.tokens_per_task,
            "n_rules": sum(1 for c in comps if c.kind == "rule"),
        },
        "stop": stop,
        "improved": float(final.train_score or 0) > float(start_train or 0),
        "note": (
            f"Stopped: {stop.get('reason')} after {stop.get('generations_run')} gen(s). "
            f"Target={stop.get('target_score')} plateau_limit={stop.get('plateau_limit')}."
        ),
    }


def _run_mocked(
    generations: int | None,
    target_score: float | None,
    plateau_limit: int | None,
) -> dict[str, Any]:
    """Mock datastore + learning apply; still uses hx.loop stop conditions."""
    os.environ["HX_USE_MOCKS"] = "1"

    from tests import mocks
    from tests.mocks import STORE

    STORE.reset_baseline()
    orig_apply = mocks.apply_edits

    def learning_apply(parent_id, edits, phase, generation):
        child = orig_apply(parent_id, edits, phase, generation)
        scores = STORE.task_scores.setdefault(child.harness_id, {})
        for tid, sc in list(scores.items()):
            if tid.startswith("train_fail_") and sc < 1.0:
                scores[tid] = 1.0
                break
        for tid in list(STORE.tasks):
            task = STORE.tasks[tid]
            if task.split == "heldout" and task.dataset == "sample_mflix":
                scores.setdefault(tid, 0.4)
                scores[tid] = max(float(scores[tid]), 0.4)
        h = STORE.harnesses[child.harness_id]
        parent = STORE.harnesses.get(parent_id)
        if phase == "prune" and parent and parent.tokens_per_task:
            STORE.harnesses[child.harness_id] = h.model_copy(
                update={"tokens_per_task": max(10.0, float(parent.tokens_per_task) - 5.0)}
            )
        return child

    mocks.apply_edits = learning_apply  # type: ignore[assignment]
    try:
        logging.basicConfig(level=logging.INFO)
        from hx.loop import evolve
        from tests.mocks import load_harness as mock_load_harness
        from tests.mocks import run_generation as mock_run_generation

        stop: dict[str, Any] = {}
        start, _ = mock_load_harness("h0")
        final_id = evolve(
            "sample_mflix",
            "h0",
            generations,
            target_score=target_score,
            plateau_limit=plateau_limit,
            stop_info=stop,
        )
        final, comps = mock_load_harness(final_id)
        measured = mock_run_generation(final_id, "sample_mflix")
        history: list[dict[str, Any]] = []
        for hid, h in sorted(STORE.harnesses.items(), key=lambda kv: (kv[1].generation, kv[0])):
            history.append(
                {
                    "harness_id": hid,
                    "generation": h.generation,
                    "phase": h.phase,
                    "train_score": h.train_score,
                    "heldout_score": h.heldout_score,
                    "tokens_per_task": h.tokens_per_task,
                    "status": h.status,
                    "parent_id": h.parent_id,
                    "n_rules": h.n_rules,
                }
            )
        curve = [
            h
            for h in history
            if h.get("status") in ("static", "promoted") and h.get("train_score") is not None
        ]
        return {
            "dataset": "sample_mflix",
            "skipped": False,
            "mode": "demo RSI loop (HX_USE_MOCKS + simulated fixes)",
            "provider": "mock",
            "start": {
                "harness_id": "h0",
                "train_score": start.train_score,
            },
            "final": {
                "harness_id": final.harness_id,
                "generation": final.generation,
                "train_score": measured.train_score,
                "heldout_score": measured.heldout_score,
                "tokens_per_task": measured.tokens_per_task,
                "n_rules": sum(1 for c in comps if c.kind == "rule"),
            },
            "curve": curve,
            "history": history,
            "stop": stop,
            "improved": float(measured.train_score or 0) > float(start.train_score or 0),
            "note": (
                f"Stopped: {stop.get('reason')} after {stop.get('generations_run')} gen(s). "
                "Set GROQ_API_KEY and HX_USE_MOCKS=0 for live Groq Llama evolution."
            ),
        }
    finally:
        mocks.apply_edits = orig_apply  # type: ignore[assignment]


def maybe_auto_evolve() -> dict[str, Any] | None:
    """Called by the ask agent — evolve only when below accuracy target."""
    ok, why = should_evolve("sample_mflix")
    if not ok:
        return {"skipped": True, "reason": why}
    return run_mflix_rsi()
