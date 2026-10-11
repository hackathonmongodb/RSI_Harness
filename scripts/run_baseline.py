#!/usr/bin/env python3
"""Run h0 and h_mongodb on mflix tasks. LLM_FAKE=1 until live keys exist."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hx.generation import run_generation  # noqa: E402
from hx.harness import H0, H_MONGODB  # noqa: E402


def _print_report(label: str, report) -> None:
    print(f"\n=== {label} ({report.harness_id}) {report.elapsed_s:.2f}s ===")
    print("train:", json.dumps(report.train_scores, indent=2))
    print("held-out:", json.dumps(report.heldout_scores, indent=2))
    print("tokens per task:", json.dumps(report.tokens_per_task, indent=2))
    print(
        json.dumps(
            {
                "n_tasks": len(report.traces),
                "cached": report.cached,
                "total_tokens": sum(report.tokens_per_task.values()),
            }
        )
    )


def main() -> None:
    persist = os.getenv("MONGODB_URI", "").strip() != ""
    t0 = time.perf_counter()
    r0 = run_generation(H0, seed=0, persist=persist)
    rm = run_generation(H_MONGODB, seed=0, persist=persist)
    _print_report("h0", r0)
    _print_report("h_mongodb", rm)
    wall = time.perf_counter() - t0
    n = len(r0.traces) + len(rm.traces)
    print(f"\nwall_clock_s={wall:.2f} tasks={n} LLM_FAKE={os.getenv('LLM_FAKE', '1')}")
    print("target: ~42-task generation in <= 120s once A's full set + live model are on")


if __name__ == "__main__":
    main()
