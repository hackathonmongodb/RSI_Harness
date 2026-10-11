"""Thin Groq (Llama) client for Stream A eval + scoring.

Uses the OpenAI SDK against https://api.groq.com/openai/v1.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

log = logging.getLogger("llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def default_model() -> str:
    return (
        os.getenv("GROQ_MODEL", "").strip()
        or os.getenv("TASK_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def _client():
    from openai import OpenAI

    key = groq_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing — set it in .env")
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def complete(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Chat completion via Groq. Retries on 429/5xx with exponential backoff."""
    use_model = (model or default_model()).strip() or DEFAULT_MODEL
    kwargs: dict[str, Any] = {
        "model": use_model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    client = _client()
    last_err: BaseException | None = None
    t0 = time.perf_counter()
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "groq model=%s prompt_tokens=%s completion_tokens=%s latency_ms=%s attempt=%s",
                use_model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
                attempt + 1,
            )
            # stash last-call stats for run_eval aggregation
            complete.last_usage = {  # type: ignore[attr-defined]
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": latency_ms,
                "model": use_model,
            }
            return text
        except Exception as err:  # noqa: BLE001
            last_err = err
            status = getattr(err, "status_code", None) or getattr(err, "status", None)
            msg = str(err).lower()
            retryable = status in {429, 500, 502, 503, 504} or "rate" in msg or "429" in msg
            if not retryable or attempt == 5:
                raise
            sleep_s = min(2**attempt, 16)
            log.warning("groq retry attempt=%s sleep=%ss err=%s", attempt + 1, sleep_s, err)
            time.sleep(sleep_s)
    raise RuntimeError(f"groq complete failed: {last_err}")


complete.last_usage = {  # type: ignore[attr-defined]
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency_ms": 0,
    "model": "",
}
