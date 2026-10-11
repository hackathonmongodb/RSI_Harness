# Part B — Runtime

SPEC §7 owners: `hx/llm.py`, `hx/harness.py`, `hx/task_agent.py`, `hx/generation.py`, `scripts/seed_baseline.py`, `scripts/run_baseline.py`.

```python
complete(model, system, messages, tools=None, temperature=0.0, max_tokens=2048) -> LLMResult
load_harness(harness_id) -> (Harness, list[Component])
assemble(harness, comps, task, ctx) -> (system, messages, tools)
apply_edits(parent_id, edits, phase, generation) -> Harness
run_task(harness_id, task, seed=0) -> Trace
run_generation(harness_id, dataset, seeds=[0], parallel=8, task_ids=None) -> GenerationResult
```
