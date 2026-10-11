"""Part B LLM adapter. Signature is the 2:00 critical-path contract for C.

complete(model, system, messages, tools, temperature, max_tokens) -> LLMResult
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from hx.models import LLMResult

FAKE_JSON = {
    "collection": "movies",
    "operation": "aggregate",
    "pipeline": [
        {"$match": {"year": 2015}},
        {"$sort": {"num_mflix_comments": -1}},
        {"$limit": 10},
        {"$project": {"_id": 0, "genres": 1, "runtime": 1}},
    ],
}


def fake_enabled() -> bool:
    return os.getenv("LLM_FAKE", "1").strip() not in {"0", "false", "False"}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _is_rate_limit(err: BaseException) -> bool:
    name = type(err).__name__.lower()
    msg = str(err).lower()
    if "rate" in name or "429" in msg or "rate_limit" in msg or "too many requests" in msg:
        return True
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    return status == 429


def complete(
    model: str | None = None,
    system: str = "",
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0,
    max_tokens: int = 1024,
) -> LLMResult:
    """One completion. Token counts are always set. Rate limits retry with backoff."""
    messages = messages or []
    tools = tools or []
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    if fake_enabled():
        text = json.dumps(FAKE_JSON)
        prompt = system + json.dumps(messages)
        return LLMResult(
            text=text,
            input_tokens=_estimate_tokens(prompt),
            output_tokens=_estimate_tokens(text),
            model="fake",
            fake=True,
            retries=0,
        )
    return _live_complete(model, system, messages, tools, temperature, max_tokens)


def _live_complete(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> LLMResult:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LLM_FAKE=0 but ANTHROPIC_API_KEY is empty")
    from anthropic import Anthropic

    client = Anthropic(api_key=key)
    anth_messages = _to_anthropic_messages(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system or "",
        "messages": anth_messages or [{"role": "user", "content": "Hello"}],
    }
    anth_tools = [_to_anthropic_tool(t) for t in tools]
    if anth_tools:
        kwargs["tools"] = anth_tools

    last_err: BaseException | None = None
    retries = 0
    for attempt in range(6):
        try:
            msg = client.messages.create(**kwargs)
            text_parts = []
            for block in msg.content:
                if getattr(block, "type", "") == "text":
                    text_parts.append(block.text)
            usage = getattr(msg, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            return LLMResult(
                text="\n".join(text_parts),
                input_tokens=in_tok,
                output_tokens=out_tok,
                model=model,
                fake=False,
                retries=retries,
            )
        except Exception as err:  # noqa: BLE001 — classify then re-raise
            last_err = err
            if not _is_rate_limit(err) or attempt == 5:
                raise
            retries += 1
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"complete failed after retries: {last_err}")


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        fn = tool["function"]
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("input_schema")
        or tool.get("parameters")
        or {"type": "object", "properties": {}},
    }


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            out.append({"role": "user", "content": f"Tool result: {m.get('content', '')}"})
            continue
        out.append(
            {
                "role": "user" if role == "user" else "assistant",
                "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")),
            }
        )
    return out
