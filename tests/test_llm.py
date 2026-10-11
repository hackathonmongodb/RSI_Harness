import pytest

from hx.llm import _parse_duration, parse_json_object


@pytest.mark.parametrize("msg,secs", [
    ("Please try again in 1.5s. Need more", 1.5),
    ("Please try again in 590ms.", 0.59),
    ("Please try again in 1m2.5s.", 62.5),
    ("try again in 7m12s", 432.0),
])
def test_parse_retry_duration(msg, secs):
    assert _parse_duration(msg) == pytest.approx(secs)


def test_parse_retry_duration_absent():
    assert _parse_duration("nothing here") is None


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


class _Err(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


class _Resp:
    class _Choice:
        class message:
            content = "hello"
            tool_calls = None
        finish_reason = "stop"

    choices = [_Choice()]

    class usage:
        prompt_tokens = 10
        completion_tokens = 2


def _llm_with(monkeypatch, outcomes):
    from hx import llm as llm_mod
    from hx.config import Settings

    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    client = llm_mod.LLM(Settings(api_key="x"))
    seen = []

    def create(**kw):
        seen.append(kw)
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    return client, seen


def test_rate_limit_waits_then_succeeds(monkeypatch):
    waits = []
    client, seen = _llm_with(monkeypatch, [_Err(429, "Rate limit. Please try again in 2.5s."), _Resp()])
    client.on_wait = lambda s, r: waits.append(s)
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r.content == "hello" and waits == [3.0] and len(seen) == 2


def test_malformed_generation_resamples_then_reports(monkeypatch):
    bad = lambda: _Err(400, "{'code': 'output_parse_failed', 'failed_generation': 'x'}")  # noqa: E731
    client, seen = _llm_with(monkeypatch, [bad(), bad(), bad()])
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r.error.startswith("malformed_generation")
    assert [kw["temperature"] for kw in seen] == [0.0, 0.3, 0.6]


def test_other_400_raises(monkeypatch):
    from hx.llm import LLMError

    client, _ = _llm_with(monkeypatch, [_Err(400, "context_length_exceeded")])
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_salvage_tool_call_with_raw_newlines():
    from hx.llm import salvage_tool_calls

    fg = '{"name": "write_file", "arguments": {"path": "t.py", "content": "def f():\n\treturn \\"x\\"\n"}}'
    calls = salvage_tool_calls(fg)
    assert len(calls) == 1 and calls[0].name == "write_file"
    assert calls[0].arguments["content"] == 'def f():\n\treturn "x"\n'
    import json
    assert json.loads(calls[0].raw_arguments)["path"] == "t.py"     # re-serialised as strict JSON
    assert salvage_tool_calls("not json at all") == []
    assert salvage_tool_calls('{"name": "bash"}') == []               # no arguments -> not a call


def test_chat_returns_salvaged_call_without_resampling(monkeypatch):
    err = _Err(400, "tool_use_failed")
    err.body = {"code": "tool_use_failed",
                "failed_generation": '{"name": "bash", "arguments": {"command": "echo\thi"}}'}
    client, seen = _llm_with(monkeypatch, [err])
    r = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "bash"}}])
    assert len(seen) == 1 and r.finish_reason == "salvaged"
    assert r.tool_calls[0].arguments == {"command": "echo\thi"}
