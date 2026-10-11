# Part B — Runtime (for C)

Owned files: `hx/llm.py`, `hx/harness.py`, `hx/task_agent.py`, `hx/generation.py`, `scripts/seed_baseline.py`, `scripts/run_baseline.py`. Branch `b/runtime`.

```
Task + Harness
    assemble()           # prompt from components + rules + policy
    complete()           # temperature 0 in run_task; LLM_FAKE canned JSON
    parse + validate     # pipeline, error-feedback retry if policy.max_retries > 0
    executor/scorer      # A; structural 0/1 until then
    Trace
run_generation()         # ThreadPoolExecutor, cache (harness_id, seed, task ids)
```

```python
complete(model, system, messages, tools, temperature, max_tokens) -> LLMResult
# LLMResult: text, input_tokens, output_tokens, model, fake, retries

assemble(harness, task) -> AssembledPrompt
apply_edits(harness, edits) -> Harness   # version + 1

run_task(harness, task, seed=0) -> Trace
run_generation(harness, seed=0, task_ids=None, persist=False) -> GenerationReport
```

`H0` vs `H_MONGODB`: same task yields different `context_components` / system prompts.

Env: `LLM_FAKE=1`, `HX_CONCURRENCY=4`. If rate-limited, lower concurrency first.
