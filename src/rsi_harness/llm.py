from __future__ import annotations

import json
from typing import Any

from rsi_harness.config import (
    anthropic_key,
    anthropic_model,
    openai_key,
    openai_model,
)


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str) -> str:
    if anthropic_key():
        return _anthropic(system, user)
    if openai_key():
        return _openai(system, user)
    raise LLMError(
        "No LLM key set. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env so "
        "natural-language → mongosh generation can run."
    )


def _anthropic(system: str, user: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=anthropic_key())
    msg = client.messages.create(
        model=anthropic_model(),
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai(system: str, user: str) -> str:
    import httpx

    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai_key()}"},
        json={
            "model": openai_model(),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise LLMError(f"Expected JSON object, got: {text[:400]}")
    return json.loads(text[start : end + 1])
