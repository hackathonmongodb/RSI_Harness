#!/usr/bin/env python3
"""Show meta user cannot write gold / held-out (SPEC §2 demo beat)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx.config import get_client


def main() -> None:
    meta = get_client("meta")
    for label, coll, filt, update in [
        (
            "gold",
            meta["hx_eval"]["gold"],
            {},
            {"$set": {"verified": True}},
        ),
        (
            "heldout_results",
            meta["hx_eval"]["heldout_results"],
            {},
            {"$set": {"score": 1.0}},
        ),
    ]:
        try:
            coll.update_one(filt, update, upsert=True)
            print(f"{label}: UNEXPECTED SUCCESS (permissions may not be enforced on Atlas tier)")
        except Exception as err:  # noqa: BLE001
            print(f"{label}: denied as expected — {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
