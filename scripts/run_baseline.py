#!/usr/bin/env python3
"""Print train / held-out / tokens-per-task for h0 and h_mongodb on mflix."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hx.generation import run_generation  # noqa: E402


def _print(label: str, r) -> None:
    print(f"\n=== {label} ===")
    print(
        json.dumps(
            {
                "harness_id": r.harness_id,
                "train_score": r.train_score,
                "heldout_score": r.heldout_score,
                "tokens_per_task": r.tokens_per_task,
                "train_outcomes": r.train_outcomes,
                "heldout_outcomes": r.heldout_outcomes,
            },
            indent=2,
        )
    )


def main() -> None:
    t0 = time.perf_counter()
    r0 = run_generation("h0", "sample_mflix", seeds=[0])
    rm = run_generation("h_mongodb", "sample_mflix", seeds=[0])
    _print("h0", r0)
    _print("h_mongodb", rm)
    print(
        json.dumps(
            {
                "wall_clock_s": round(time.perf_counter() - t0, 2),
                "LLM_FAKE": os.getenv("LLM_FAKE", "0"),
            }
        )
    )


if __name__ == "__main__":
    main()
