from hx.generation import clear_cache, run_generation
from hx.harness import H0, H_MONGODB
from hx.models import Task
from hx.pipeline import parse_json_answer
from hx.task_agent import run_task


def test_run_task_populates_trace(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    task = Task(task_id="t1", nl="Show 10 movies from 2015", split="train")
    tr = run_task(H_MONGODB, task, seed=0)
    assert tr.harness_id == "h_mongodb"
    assert tr.context_components
    assert "schema" in tr.context_components
    assert tr.input_tokens > 0
    assert tr.output_tokens > 0
    assert tr.parsed is not None
    assert tr.pipeline
    assert tr.score == 1.0
    parse_json_answer(tr.raw_response)


def test_run_generation_splits_and_cache(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    clear_cache()
    r1 = run_generation(H0, seed=0, persist=False)
    r2 = run_generation(H0, seed=0, persist=False)
    assert r2.cached is True
    assert set(r1.train_scores) | set(r1.heldout_scores) == set(r1.tokens_per_task)
    assert r1.elapsed_s >= 0
