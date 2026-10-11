from __future__ import annotations

import json
from typing import Any

from rsi_harness.config import groq_key, groq_model


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str) -> str:
    key = groq_key()
    if not key:
        raise LLMError("No LLM key set. Add GROQ_API_KEY to .env.")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=groq_model(),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or ""
    return content


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
