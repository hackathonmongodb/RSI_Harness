import json
import os

from hx.llm import FAKE_JSON, complete, fake_enabled


def test_fake_complete_signature_and_tokens(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    assert fake_enabled()
    result = complete(
        "claude-sonnet-4-5",
        "sys",
        [{"role": "user", "content": "Show movies from 2015"}],
        None,
        0,
        256,
    )
    assert result.fake is True
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    parsed = json.loads(result.text)
    assert parsed == FAKE_JSON
    assert parsed["operation"] == "aggregate"
    assert isinstance(parsed["pipeline"], list)


def test_fake_json_is_valid_answer():
    from hx.pipeline import parse_json_answer, pipeline_from_parsed, validate_pipeline

    parsed = parse_json_answer(json.dumps(FAKE_JSON))
    pipe = pipeline_from_parsed(parsed)
    validate_pipeline(pipe)
