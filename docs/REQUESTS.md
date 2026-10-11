# Requests from B (runtime)

Aligned with SPEC.md §7. B will not edit A/C-owned modules.

## A

1. `hx/config.py` `get_client(role)` with real Atlas URIs in `.env`.
2. `hx/executor.py`: `validate_pipeline`, `run_pipeline`.
3. `hx/scorer.py`: `normalize`, `score(task, parsed) -> (float, ErrorType, str|None)`, `get_noise_floor`, `load_tasks`.
4. `hx/schema_probe.py`: `get_dataset_context`.
5. ~30 train + 12 held-out mflix tasks + gold.

Until those land, B uses in-memory `h0`/`h_mongodb`, stub schema context, and structural 0/1 scores.

## C

```python
from hx.llm import complete, LLMResult
from hx.harness import load_harness, assemble, apply_edits
from hx.task_agent import run_task
from hx.generation import run_generation
from hx.models import Edit, Component, Harness, Trace, GenerationResult
```

`complete(model, system, messages, tools=None, temperature=0.0, max_tokens=2048) -> LLMResult`
with `prompt_tokens`, `completion_tokens`, `tool_calls`. `LLM_FAKE=1` returns canned `{"collection","pipeline"}`.
