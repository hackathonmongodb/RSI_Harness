"""Stub until A owns this. SPEC: get_client(role) -> MongoClient."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Role = Literal["admin", "task", "meta"]
_ENV = {
    "admin": "MONGODB_URI_ADMIN",
    "task": "MONGODB_URI_TASK",
    "meta": "MONGODB_URI_META",
}


def mongodb_uri(role: Role = "admin") -> str:
    return (os.getenv(_ENV[role]) or os.getenv("MONGODB_URI") or "").strip()


@lru_cache(maxsize=8)
def get_client(role: Role = "admin"):
    uri = mongodb_uri(role)
    if not uri:
        raise RuntimeError(f"no URI for role={role}; A fills env.example / .env")
    from pymongo import MongoClient

    return MongoClient(uri, serverSelectionTimeoutMS=8000)
