from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def mongodb_uri() -> str:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "Set MONGODB_URI in .env to your Atlas Sandbox connection string."
        )
    return uri


def mongodb_db() -> str:
    return os.getenv("MONGODB_DB", "rsi_harness")


def anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
