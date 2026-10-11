#!/usr/bin/env python3
"""Run h0 three times once B's run_generation works; store std + margin in hx_eval.noise."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx.config import get_client


def main(dataset: str = "sample_mflix", harness_id: str = "h0") -> None:
    from hx.generation import run_generation

    scores: list[float] = []
    for seed in (0, 1, 2):
        result = run_generation(harness_id, dataset, seeds=[seed])
        scores.append(float(result.train_score))
        print(f"seed={seed} train_score={result.train_score}")

    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    margin = 2.0 * std
    get_client("admin")["hx_eval"]["noise"].update_one(
        {"dataset": dataset},
        {
            "$set": {
                "dataset": dataset,
                "harness_id": harness_id,
                "scores": scores,
                "std": std,
                "margin": margin,
            }
        },
        upsert=True,
    )
    print(f"stored noise std={std:.4f} margin(2*std)={margin:.4f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "sample_mflix"
    main(ds)
