"""Read-only access to Atlas sample_mflix (movie database)."""

from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from rsi_harness.config import mongodb_uri


@lru_cache(maxsize=1)
def mflix_client() -> MongoClient:
    kwargs: dict = {"serverSelectionTimeoutMS": 8000}
    try:
        import certifi

        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        pass
    return MongoClient(mongodb_uri(), **kwargs)


def mflix() -> Database:
    return mflix_client()["sample_mflix"]
