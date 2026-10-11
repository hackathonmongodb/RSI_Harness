from hx.fallback_tasks import FALLBACK_TASKS
from hx.generation import clear_cache, run_generation
from hx.task_agent import run_task


def test_run_task_populates_trace(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    task = FALLBACK_TASKS[0]
    tr = run_task("h_mongodb", task, seed=0)
    assert tr.harness_id == "h_mongodb"
    assert tr.context_components
    assert tr.prompt_tokens > 0
    assert tr.completion_tokens > 0
    assert tr.parsed is not None
    assert tr.parsed.get("pipeline")
    assert tr.score == 1.0


def test_run_generation_splits_and_cache(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    clear_cache()
    r1 = run_generation("h0", "sample_mflix", seeds=[0])
    r2 = run_generation("h0", "sample_mflix", seeds=[0])
    assert r2.train_score == r1.train_score
    assert r1.train_outcomes
    assert r1.heldout_outcomes
    assert r1.tokens_per_task > 0
