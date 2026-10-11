"""Thin config. A owns Atlas wiring; B only needs this import to not crash.

When MONGODB_URI is unset, get_client() raises — tests never call it.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def mongodb_uri() -> str:
    return os.getenv("MONGODB_URI", "").strip()


def mongodb_db() -> str:
    return os.getenv("MONGODB_DB", "rsi_harness")


@lru_cache(maxsize=1)
def get_client():
    uri = mongodb_uri()
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is empty. Part A creates the Atlas sandbox and .env."
        )
    from pymongo import MongoClient

    return MongoClient(uri, serverSelectionTimeoutMS=8000)
