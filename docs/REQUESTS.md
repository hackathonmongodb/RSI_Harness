# Requests from B (runtime)

B will not edit files A or C own. Please implement these on your branches.

## A — Atlas / executor / scorer / tasks

1. Own `hx/config.py` for real (`get_client()`, final `.env.example`).
2. Add `hx/executor.py` with:
   ```python
   class executor:
       @staticmethod
       def execute(task: hx.models.Task, pipeline: list[dict]) -> Any: ...
   ```
   Run the parsed pipeline against `sample_mflix` (read-only).
3. Add `hx/scorer.py` with:
   ```python
   class scorer:
       @staticmethod
       def score(task, parsed, exec_result) -> float: ...
   ```
   Return 0–1 using gold / result overlap. Until this lands, B uses a structural 0/1 score.
4. Add `hx/tasks.py` with `TASKS: list[Task]` (~42 mflix NL queries, `split` in `{train, heldout}`). Until then B uses `hx/fallback_tasks.py` (5 items).
5. Cluster + three users + sample data; tell B when `MONGODB_URI` is in `.env` so `scripts/seed_baseline.py` can run.

## C — evolve

Import from B:

```python
from hx.llm import complete          # complete(model, system, messages, tools, temperature, max_tokens) -> LLMResult
from hx.harness import assemble, apply_edits, H0, H_MONGODB
from hx.task_agent import run_task   # -> Trace (context_components, token counts)
from hx.generation import run_generation
from hx.models import Edit, Harness, Policy, Trace, LLMResult
```

`LLM_FAKE=1` returns canned valid JSON. Do not fork `complete`.

Edits B supports today: `delete_rule`, `add_rule`, `set_policy` (one knob), `set_component_enabled`.
