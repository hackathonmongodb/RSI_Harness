"""Part B LLM adapter. SPEC §7 signature — do not fork this in C.

Live backend: Groq OpenAI-compatible API (Llama models).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

log = logging.getLogger("hx.llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


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


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def default_model() -> str:
    return (
        os.getenv("GROQ_MODEL", "").strip()
        or os.getenv("TASK_MODEL", "").strip()
        or DEFAULT_MODEL
    )


# Back-compat alias for earlier Llama wiring
def llama_api_key() -> str:
    return groq_api_key()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _is_rate_limit(err: BaseException) -> bool:
    msg = str(err).lower()
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    return status == 429 or "rate_limit" in msg or "too many requests" in msg or "429" in msg


def _is_retryable(err: BaseException) -> bool:
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    return _is_rate_limit(err) or "server" in str(err).lower()


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
    key = groq_api_key()
    if not key:
        raise RuntimeError("LLM_FAKE=0 but GROQ_API_KEY is empty — set it in .env")

    use_model = (model or "").strip() or default_model()
    if use_model.startswith("claude") or use_model.startswith("fake") or use_model.startswith("Llama-4"):
        use_model = default_model()

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    oai_messages = _to_openai_messages(system, messages)
    kwargs: dict[str, Any] = {
        "model": use_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": oai_messages or [{"role": "user", "content": "Hello"}],
    }
    oai_tools = [_to_openai_tool(t) for t in tools]
    if oai_tools:
        kwargs["tools"] = oai_tools

    last_err: BaseException | None = None
    retries = 0
    t0 = time.perf_counter()
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            message = resp.choices[0].message
            text = (message.content or "").strip()
            tool_calls: list[dict] = []
            for tc in message.tool_calls or []:
                fn = tc.function
                args_raw = fn.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except json.JSONDecodeError:
                    args = {"raw": args_raw}
                tool_calls.append({"id": tc.id or "", "name": fn.name or "", "args": args})
            usage = resp.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "groq model=%s prompt_tokens=%s completion_tokens=%s latency_ms=%s",
                use_model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            )
            return LLMResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_calls=tool_calls,
                model=use_model,
                retries=retries,
            )
        except Exception as err:  # noqa: BLE001
            last_err = err
            if not _is_retryable(err) or attempt == 5:
                raise
            retries += 1
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"complete failed after retries: {last_err}")


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema")
            or tool.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role") or "user"
        if role == "system":
            content = m.get("content", "")
            if content:
                out.append(
                    {
                        "role": "system",
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                )
            continue
        if role == "tool":
            out.append({"role": "user", "content": f"Tool result: {m.get('content', '')}"})
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        out.append({"role": "user" if role == "user" else "assistant", "content": content})
    return out
