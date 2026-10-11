from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from rsi_harness.config import mongodb_db, mongodb_uri

COLLECTIONS = (
    "harness_versions",
    "eval_runs",
    "memories",
    "query_traces",
    "agent_events",
)


@lru_cache(maxsize=1)
def client() -> MongoClient:
    return MongoClient(mongodb_uri(), serverSelectionTimeoutMS=8000)


def db() -> Database:
    return client()[mongodb_db()]


def ping() -> dict:
    return client().admin.command("ping")


def ensure_indexes() -> None:
    d = db()
    d.harness_versions.create_index("version", unique=True)
    d.harness_versions.create_index([("status", 1), ("created_at", -1)])
    d.eval_runs.create_index([("harness_version", 1), ("score", -1)])
    d.eval_runs.create_index([("created_at", -1)])
    d.memories.create_index([("tier", 1), ("created_at", -1)])
    d.query_traces.create_index([("created_at", -1)])
    d.agent_events.create_index([("created_at", -1)])
