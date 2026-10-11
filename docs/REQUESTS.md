# Cross-workstream requests

## From B (runtime)

Aligned with SPEC.md §7. B will not edit A/C-owned modules.

### A

1. `hx/config.py` `get_client(role)` with real Atlas URIs in `.env`.
2. `hx/executor.py`: `validate_pipeline`, `run_pipeline`.
3. `hx/scorer.py`: `normalize`, `score(task, parsed) -> (float, ErrorType, str|None)`, `get_noise_floor`, `load_tasks`.
4. `hx/schema_probe.py`: `get_dataset_context`.
5. ~30 train + 12 held-out mflix tasks + gold.

Until those land, B uses in-memory `h0`/`h_mongodb`, stub schema context, and structural 0/1 scores.

### C

```python
from hx.llm import complete, LLMResult
from hx.harness import load_harness, assemble, apply_edits
from hx.task_agent import run_task
from hx.generation import run_generation
from hx.models import Edit, Component, Harness, Trace, GenerationResult
```

`complete(model, system, messages, tools=None, temperature=0.0, max_tokens=2048) -> LLMResult`
with `prompt_tokens`, `completion_tokens`, `tool_calls`. `LLM_FAKE=1` returns canned `{"collection","pipeline"}`.

## From C (evolution)

B’s `run_generation` on main already fills `train_outcomes` and honors `task_ids=`. C uses those when `HX_USE_MOCKS` is unset.

B’s `LLM_FAKE` canned payload is a task-agent pipeline, not debugger/evolve JSON. C tests set `HX_USE_MOCKS=1` and use `tests.mocks.complete` for structured meta JSON. Live evolve uses `hx.llm.complete`.

### A

- `get_noise_floor(dataset)` should return the **promotion margin** (`2 * std`), not raw std. C uses `DEFAULT_NOISE = 0.05` until this exists. Please document which.
- `load_tasks(dataset, split)` so C can join train questions without gold.
