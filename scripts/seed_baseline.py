#!/usr/bin/env python3
"""Write h0 and h_mongodb to Atlas. Requires A's cluster + MONGODB_URI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hx.config import get_client, mongodb_db  # noqa: E402
from hx.harness import H0, H_MONGODB  # noqa: E402


def main() -> None:
    db = get_client()[mongodb_db()]
    n = 0
    for h in (H0, H_MONGODB):
        db.harnesses.update_one(
            {"harness_id": h.harness_id, "version": h.version},
            {"$set": h.model_dump()},
            upsert=True,
        )
        n += 1
    print(json.dumps({"wrote": n, "ids": ["h0", "h_mongodb"], "db": db.name}))


if __name__ == "__main__":
    main()
