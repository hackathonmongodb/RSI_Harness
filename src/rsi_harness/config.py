from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def mongodb_uri() -> str:
    uri = (
        os.getenv("MONGODB_URI", "").strip()
        or os.getenv("MONGODB_URI_ADMIN", "").strip()
        or os.getenv("MONGODB_URI_TASK", "").strip()
    )
    if not uri:
        raise RuntimeError(
            "Set MONGODB_URI (or MONGODB_URI_ADMIN) in .env to your Atlas connection string."
        )
    return uri


def mongodb_db() -> str:
    return os.getenv("MONGODB_DB", "rsi_harness")


def groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def groq_model() -> str:
    return (
        os.getenv("GROQ_MODEL", "").strip()
        or os.getenv("TASK_MODEL", "").strip()
        or "llama-3.3-70b-versatile"
    )


def llama_key() -> str:
    """Back-compat — prefer GROQ_API_KEY."""
    return groq_key() or os.getenv("LLAMA_API_KEY", "").strip()


def llama_model() -> str:
    return groq_model()


def anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def voyage_api_key() -> str:
    return (
        os.getenv("VOYAGE_API_KEY", "").strip()
        or os.getenv("MONGODB_AI_API_KEY", "").strip()
    )


def mongodb_ai_endpoint() -> str:
    return (
        os.getenv("MONGODB_AI_ENDPOINT", "").strip()
        or "https://ai.mongodb.com/v1"
    )
