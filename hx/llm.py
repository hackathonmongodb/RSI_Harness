"""Part B LLM adapter. SPEC §7 signature — do not fork this in C."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[dict] = Field(default_factory=list)
    model: str = ""
    fake: bool = False
    retries: int = 0


FAKE_JSON = {
    "collection": "movies",
    "pipeline": [
        {"$match": {"year": 2015}},
        {"$sort": {"num_mflix_comments": -1}},
        {"$limit": 10},
        {"$project": {"_id": 0, "genres": 1, "runtime": 1}},
    ],
}


def fake_enabled() -> bool:
    return os.getenv("LLM_FAKE", "0").strip() in {"1", "true", "True"}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _is_rate_limit(err: BaseException) -> bool:
    msg = str(err).lower()
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    return status == 429 or "rate_limit" in msg or "too many requests" in msg or "429" in msg


def complete(
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> LLMResult:
    tools = tools or []
    if fake_enabled():
        text = json.dumps(FAKE_JSON)
        prompt = system + json.dumps(messages)
        return LLMResult(
            text=text,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(text),
            model="fake",
            fake=True,
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
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system or "",
        "messages": _to_anthropic_messages(messages) or [{"role": "user", "content": "Hello"}],
    }
    anth_tools = [_to_anthropic_tool(t) for t in tools]
    if anth_tools:
        kwargs["tools"] = anth_tools

    last_err: BaseException | None = None
    retries = 0
    for attempt in range(6):
        try:
            msg = client.messages.create(**kwargs)
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in msg.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_use":
                    tool_calls.append(
                        {"id": block.id, "name": block.name, "args": dict(block.input or {})}
                    )
            usage = getattr(msg, "usage", None)
            return LLMResult(
                text="\n".join(text_parts),
                prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                tool_calls=tool_calls,
                model=model,
                retries=retries,
            )
        except Exception as err:  # noqa: BLE001
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
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        out.append({"role": "user" if role == "user" else "assistant", "content": content})
    return out
