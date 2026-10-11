"""OpenAI-compatible chat client (Groq by default) with tool calling and rate-limit waits.

Everything that talks to a model goes through `LLM`. Tests pass a scripted fake with the
same two methods (`chat`, `json`).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hx.config import Settings

WaitHook = Callable[[float, str], None]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    error: str = ""          # set when the provider rejected a malformed tool call


class LLMError(RuntimeError):
    pass


_RETRY_IN = re.compile(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+(?:ms|s))?)", re.I)


def _parse_duration(text: str) -> Optional[float]:
    """'1m2.5s' / '590ms' / '7.2s' -> seconds."""
    m = _RETRY_IN.search(text or "")
    if not m or not m.group(1):
        return None
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(h|ms|m|s)", m.group(1)):
        v = float(num)
        total += {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit] * v
    return total or None


def retry_after_seconds(err: BaseException) -> Optional[float]:
    resp = getattr(err, "response", None)
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return _parse_duration(str(err))


def _status(err: BaseException) -> Optional[int]:
    return getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)


def _is_malformed_generation(msg: str) -> bool:
    # Groq: tool_use_failed (bad tool-call JSON), output_parse_failed (gpt-oss harmony output unparsable)
    return "tool_use_failed" in msg or "output_parse_failed" in msg or "failed_generation" in msg


def _failed_generation(err: BaseException) -> str:
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        fg = body.get("failed_generation") or (body.get("error") or {}).get("failed_generation")
        if isinstance(fg, str):
            return fg
    return ""


def salvage_tool_calls(failed_generation: str) -> list[ToolCall]:
    """Groq rejects tool calls whose arguments are not strict JSON (gpt-oss often puts raw newlines or
    control characters inside long string arguments). Its `failed_generation` usually still holds a
    usable {"name", "arguments"} object; parse it leniently instead of spending a resample."""
    text = (failed_generation or "").strip()
    if not text:
        return []
    data: Any = None
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate, strict=False)
            break
        except json.JSONDecodeError:
            continue
    items = data if isinstance(data, list) else [data]
    calls: list[ToolCall] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if "arguments" not in fn and "parameters" not in fn:
            continue
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and name and isinstance(args, dict):
            calls.append(ToolCall(id=f"salvaged_{int(time.time() * 1000)}_{i}", name=name,
                                  arguments=args, raw_arguments=json.dumps(args)))
    return calls


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerant JSON-object extraction (code fences, leading prose)."""
    blob = (text or "").strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    try:
        val = json.loads(blob)
    except json.JSONDecodeError:
        start, end = blob.find("{"), blob.rfind("}")
        if start < 0 or end <= start:
            raise
        val = json.loads(blob[start : end + 1])
    if not isinstance(val, dict):
        raise ValueError("expected a JSON object")
    return val


class LLM:
    def __init__(self, settings: Settings, on_wait: Optional[WaitHook] = None):
        if not settings.api_key:
            raise LLMError("No API key. Set GROQ_API_KEY (or HX_API_KEY) in .env")
        from openai import OpenAI

        self.s = settings
        self.on_wait = on_wait
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0, timeout=180)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------ core
    def _create(self, *, salvage: bool = False, **kwargs):
        """Call the API with waits/retries. With salvage=True a rejected-but-recoverable tool call
        comes back as a list[ToolCall] instead of a response."""
        last: BaseException | None = None
        temperature = kwargs.get("temperature", 0.0)
        malformed = 0
        for attempt in range(10):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as err:  # noqa: BLE001
                last = err
                status = _status(err)
                msg = str(err)
                if status == 429 or "rate_limit" in msg:
                    wait = min((retry_after_seconds(err) or 5.0) + 0.5, 120.0)
                    if self.on_wait:
                        self.on_wait(wait, "rate limit")
                    time.sleep(wait)
                    continue
                if status == 400 and _is_malformed_generation(msg):
                    if salvage:
                        calls = salvage_tool_calls(_failed_generation(err))
                        if calls:
                            return calls
                    if malformed < 2:
                        # Model emitted unparsable output / tool call: resample a little warmer.
                        malformed += 1
                        temperature = min(1.0, temperature + 0.3)
                        kwargs["temperature"] = temperature
                        continue
                    raise LLMError("malformed_generation: " + msg[:500]) from err
                if status in (500, 502, 503, 504) or "timed out" in msg.lower() or "connection" in msg.lower():
                    wait = min(2 ** attempt, 30)
                    if self.on_wait:
                        self.on_wait(wait, f"provider error {status or ''}".strip())
                    time.sleep(wait)
                    continue
                raise LLMError(msg[:800]) from err
        raise LLMError(f"gave up after retries: {last}")

    def _extra(self, model: str) -> dict[str, Any]:
        if self.s.reasoning_effort and "gpt-oss" in model:
            return {"reasoning_effort": self.s.reasoning_effort}
        return {}

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Reply:
        use = model or self.s.model
        kwargs: dict[str, Any] = {
            "model": use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        extra = self._extra(use)
        if extra:
            kwargs["extra_body"] = extra
        try:
            resp = self._create(salvage=bool(tools), **kwargs)
        except LLMError as err:
            if str(err).startswith("malformed_generation"):
                return Reply(model=use, error=str(err))
            raise
        if isinstance(resp, list):   # tool call recovered from the provider's failed_generation
            return Reply(tool_calls=resp, model=use, finish_reason="salvaged")
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if not isinstance(args, dict):
                    args = {"_value": args}
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            calls.append(ToolCall(id=tc.id or f"call_{len(calls)}", name=tc.function.name or "", arguments=args, raw_arguments=raw))
        usage = resp.usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        return Reply(
            content=(msg.content or "").strip(),
            tool_calls=calls,
            prompt_tokens=pt,
            completion_tokens=ct,
            model=use,
            finish_reason=choice.finish_reason or "",
        )

    def json(self, system: str, user: str, *, model: Optional[str] = None, max_tokens: int = 1500) -> dict[str, Any]:
        """Side-call that must return one JSON object (recall, reflection)."""
        use = model or self.s.model
        kwargs: dict[str, Any] = {
            "model": use,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        extra = self._extra(use)
        if extra:
            kwargs["extra_body"] = extra
        resp = self._create(**kwargs)
        usage = resp.usage
        self.total_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return parse_json_object(resp.choices[0].message.content or "")
