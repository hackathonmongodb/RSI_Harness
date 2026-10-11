#!/usr/bin/env python3
"""Write h0 and h_mongodb component + harness docs to hx_harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hx.config import get_client  # noqa: E402
from hx.harness import load_harness, seed_baselines  # noqa: E402


def main() -> None:
    seed_baselines()
    db = get_client("meta")["hx_harness"]
    n = 0
    for hid in ("h0", "h_mongodb"):
        h, comps = load_harness(hid)
        for c in comps:
            db.components.update_one(
                {"component_id": c.component_id, "version": c.version},
                {"$set": c.model_dump()},
                upsert=True,
            )
            n += 1
        db.harnesses.update_one(
            {"harness_id": h.harness_id},
            {"$set": h.model_dump()},
            upsert=True,
        )
    print(json.dumps({"components_upserted": n, "harnesses": ["h0", "h_mongodb"]}))


if __name__ == "__main__":
    main()
