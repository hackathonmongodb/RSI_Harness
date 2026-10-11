"""Atlas clients by role. SPEC §7 / PROMPTS Person A."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Explicit path so scripts and notebooks always find the project .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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
    """Return a MongoClient for admin | task | meta from .env."""
    uri = mongodb_uri(role)
    if not uri:
        raise RuntimeError(
            f"no URI for role={role}; set {_ENV[role]} in .env (never commit .env)"
        )
    from pymongo import MongoClient

    return MongoClient(uri, serverSelectionTimeoutMS=8000)
