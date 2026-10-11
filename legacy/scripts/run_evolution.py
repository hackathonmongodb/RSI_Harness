"""CLI: python -m scripts.run_evolution --dataset sample_mflix --gens 10 [--start HARNESS_ID]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx.loop import evolve


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sample_mflix")
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--start", default="h0", help="Harness id to resume from (h0 = cold start)")
    args = p.parse_args(argv)
    hid = evolve(args.dataset, args.start, args.gens)
    print(hid)
    return hid


if __name__ == "__main__":
    main()
