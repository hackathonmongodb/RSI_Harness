# PROMPTS.md — paste-ready Cursor prompts, one workstream per person

## How to run this (the "smarter way")
Three people already give you three parallel streams, so don't spin up many agents per person. Split by FILE OWNERSHIP behind a frozen contract (SPEC.md + hx/models.py), one Cursor agent session per person, plan mode first. Use subagents inside a session only for independent files like tests and scripts.

Do this before anyone prompts anything (Person A, about 15 minutes):
1. Create the repo (public), add `AGENTS.md`, `SPEC.md`, `PROMPTS.md`, and `models.py` as `hx/models.py`. Add `.gitignore` (`.env`), `.env.example`, `docs/REQUESTS.md`, empty `hx/__init__.py`. Push to `main`. B and C clone and branch.
2. Create the project + cluster from the Atlas sandbox email link. Create DB users `admin`, `task`, `meta` per SPEC section 2. Load the sample datasets. Share `.env` over DM, not git.
3. Everyone installs MongoDB Agent Skills and connects the MongoDB MCP server to the sandbox cluster in Cursor (use the `task` user's URI for the MCP server so it is read-only).

Then everyone pastes the KICKOFF prompt with their letter, and answers questions from plan mode before approving.

---

## KICKOFF (everyone pastes this first, filling in the letter)

```text
You are workstream {A|B|C} of a 3-person hackathon project that has to be submitted by 5:00 PM today. Read AGENTS.md, SPEC.md and hx/models.py, then the section for workstream {A|B|C} in PROMPTS.md.

Stay in plan mode. Output:
1. The files you will create, each with the public function signatures exactly as in SPEC.md section 7.
2. The order you will build them in, with the test that proves each one works.
3. Anything in SPEC.md that is ambiguous, inconsistent or wrong. Do not fix it silently; list it.
4. Which sub-tasks are independent enough to hand to parallel subagents (tests, scripts), and which files each would own.
Then wait for my approval.

After approval: implement in the agreed order, run the module's tests, commit to my branch and push after each module. Never edit files owned by another workstream; append requests to docs/REQUESTS.md. Never change a signature or a model without telling me first.
```

---

## Person A — Eval & Data (branch `a/eval`)

**Owns:** `hx/config.py`, `hx/executor.py`, `hx/scorer.py`, `hx/schema_probe.py`, `scripts/init_db.py`, `scripts/load_samples.py`, `scripts/make_tasks.py`, `scripts/noise_floor.py`, `scripts/demo_denied.py`, `dashboard/`.
Whoever has the Atlas sandbox email should be A. A is front-loaded, then moves to the dashboard around 2:00.

```text
You are Person A: Eval & Data. You own the ground truth and the database setup; if scoring is wrong, everything downstream is noise.

Build in this order:
1. hx/config.py: get_client(role) for admin/task/meta from env vars MONGODB_URI_ADMIN / _TASK / _META. .env.example with every variable.
2. scripts/init_db.py: create hx_eval, hx_evidence, hx_harness collections, indexes (traces by harness_id+task_id, reports by trace_id and implicated_component_id, components by component_id+version, harnesses by parent_id), and $jsonSchema validators per SPEC section 2 (components: kind enum, content maxLength 400, status enum; manifests: non-empty root_cause/targeted_fix, predicted_fixes present). Write a test that an invalid component is rejected by the database.
3. hx/executor.py: validate_pipeline (block $out $merge $function $accumulator $where, max 12 stages, must be a list of dicts) and run_pipeline using role "task" with maxTimeMS and a row cap. Unit tests for each blocked stage.
4. hx/scorer.py: normalize + score + get_noise_floor per SPEC section 3. Tests must cover Decimal128 vs float, Int64 vs int, ObjectId, datetime, key order, ordered vs unordered, extra fields, empty result. Gold is read via the admin client from hx_eval.gold.
5. hx/schema_probe.py: get_dataset_context(dataset) builds and caches (hx_harness.dataset_context) a description, annotated TypeScript schema with a comment on every field (generate ONCE per collection with META_MODEL from 3 sample docs; do not regenerate), up to 3 sample docs per collection serialized with bson.json_util in relaxed EJSON, untruncated (B truncates by policy), and index descriptions. load_tasks(dataset, split).
6. scripts/make_tasks.py: use the MongoDB MCP server to inspect sample_mflix first. Have META_MODEL draft question + gold pipeline pairs (about 30 train, 12 held-out), execute each gold, drop errors and empty results, tag by difficulty and category (decimal128, date, lookup, array, nested, group), and write to hx_eval.tasks and hx_eval.gold. Every question names the exact output fields and any order/limit and uses absolute dates. I will hand-verify at least 10; add a script flag to print a task, its gold pipeline and its first 3 result rows for review. Then repeat for sample_analytics (inspect the real fields first; it needs more $lookup).
7. scripts/noise_floor.py: once B's run_generation exists, run h0 three times, store the std in hx_eval.noise.
8. scripts/demo_denied.py: using the meta client, attempt to update a gold doc and a held-out result, and print the Unauthorized errors.
9. From about 2:00: dashboard/app.py per SPEC section 6 (Streamlit, st.graphviz_chart for lineage, $graphLookup for lineage queries, $setWindowFields for moving averages).

Definition of done for steps 1-6 by 12:30-1:30: pytest passes; `python -c "from hx.schema_probe import load_tasks; print(len(load_tasks('sample_mflix','train')))"` prints about 30.
```

---

## Person B — Runtime (branch `b/runtime`)

**Owns:** `hx/llm.py`, `hx/harness.py`, `hx/task_agent.py`, `hx/generation.py`, `scripts/seed_baseline.py`, `scripts/run_baseline.py`.
B pushes `llm.py` within 20 minutes because C needs its signature. B is the critical path for the 2:00 integration checkpoint.

```text
You are Person B: Runtime. You own the thing being evolved: assembling a harness from component documents, running the task agent, and running a whole generation fast.

Build in this order:
1. hx/llm.py FIRST, push within 20 minutes. complete(model, system, messages, tools, temperature, max_tokens) -> LLMResult. Provider-agnostic wrapper over the SDK we use, with retry/backoff on rate limits and token counts on every result. Model names from env TASK_MODEL and META_MODEL. Add a FAKE mode (env LLM_FAKE=1) returning a canned valid JSON answer so C and A can test without spending credits.
2. scripts/seed_baseline.py: seed two static harnesses per SPEC section 4 into hx_harness via the admin client: h0 (rule.base pinned + default policies) and h_mongodb (the nine general tips as nine rules, CoT on, 3 sample docs, ts_annotated schema, indexes on, 3 few-shot rules written by you in the style of the MongoDB guide, using only mflix train questions that are NOT in the held-out split). Component ids: rule.base, rule.tip_1..9, rule.fewshot_1..3, policy.<key>. Compute token_cost as len(text)/4.
3. hx/harness.py: load_harness, assemble (exact shape in SPEC section 4: system prompt + database block in the MongoDB-recommended layout), apply_edits (writes new component versions and a child Harness doc with status candidate, n_rules / n_tools set, parent_id set, previous versions untouched). Unit tests: assembling h0 vs h_mongodb produces different prompts; deleting a rule removes exactly its text; modifying a policy changes only that knob.
4. hx/task_agent.py: run_task builds the prompt, calls TASK_MODEL at temperature 0, parses the JSON (tolerate code fences), expands tool calls (stretch), runs executor.validate_pipeline, runs the pipeline for the error-feedback retry loop if policy.max_retries > 0 (feed the error text back once per retry), calls scorer.score, and returns a fully populated Trace including context_components, tool_calls, token counts, latency. If policy.sample_docs_tool is on, expose sample_docs(collection, n) as a real tool.
5. hx/generation.py: run_generation with ThreadPoolExecutor, idempotent and cached per (harness_id, seed, task set). Train traces go to hx_evidence.traces; held-out only writes hx_eval.heldout_results (score + tokens, no outputs). Update the harness doc's train_score, heldout_score, tokens_per_task. Return GenerationResult with failed/passed train trace ids. The task_ids argument restricts the run to a subset (used by ablation).
6. scripts/run_baseline.py: runs h0 and h_mongodb on sample_mflix and prints train / held-out / tokens-per-task. This is the 2:00 integration milestone.
7. Stretch, only after C's loop works: kind="tool" support (expand {"tool", "args"} into a pipeline by textual param substitution into EJSON, then validate).

Watch: concurrency vs rate limits; a full generation (42 tasks) must be about 2 minutes or less. Measure it and tell the team the number.
```

---

## Person C — Evolution (branch `c/evolve`)

**Owns:** `hx/debugger.py`, `hx/overview.py`, `hx/evolve.py`, `hx/ablate.py`, `hx/promote.py`, `hx/analytics.py`, `hx/loop.py`, `scripts/run_evolution.py`.
C is the riskiest stream: give it to whoever is strongest at LLM-eval and agent-loop design. C develops against mocks for the first ~90 minutes.

```text
You are Person C: Evolution. You own the meta-loop: turning failure evidence into falsifiable edits and deciding what survives. The paper's idea (Agentic Harness Engineering) is that every edit must be grounded in evidence and carry a prediction; our additions are pruning and prediction scoring.

Start by writing tests/mocks.py: fake implementations of run_generation, load_harness, apply_edits, score and get_noise_floor matching SPEC section 7 signatures exactly (use hx/llm.py's LLM_FAKE mode when B pushes it), so you can build against them until A and B land. Then build in this order:
1. hx/debugger.py: analyze(harness_id, trace_ids) -> list[Report]. For each trace, the META_MODEL (or a cheaper one) gets the question, the harness's component list with ids, the model's output and the error type / execution diff, and returns structured JSON: root_cause, implicated_component_id OR missing_component=true, confidence, short evidence excerpt. Parallelize. Run on all failed train traces plus up to 5 passes. Write reports via the admin client path owned by the orchestrator (do not give the debugger LLM a DB handle). Validate JSON against models.Report and retry once on invalid output.
2. hx/overview.py: build_overview(harness_id) using one aggregation with $facet: counts by implicated_component_id, implicated_kind, error_type, missing_component, and top failing tags.
3. hx/evolve.py: propose_edits(harness_id, phase, overview) -> list[Manifest]. The meta-agent sees the overview first and fetches individual reports/traces by id only through a function you expose; TRAIN split only. It must output structured JSON validated against models.Edit and models.Manifest: op, target_component_id, root_cause, targeted_fix, evidence_trace_ids, predicted_fixes, predicted_regressions (train task ids). Enforce in code: never touch pinned components; policy edits only on existing keys within the SPEC range; rule content <= 400 chars; respect per-phase caps (policy <= 2, growth <= 3, prune <= 3); reject edits that name no evidence. Few-shot rules should be built from passing train traces' own outputs, never from gold.
4. hx/ablate.py: prune_candidates(harness_id, dataset): for each active non-pinned rule/tool, build a candidate via apply_edits(delete), run run_generation(task_ids=train tasks where the component appears in context_components), compute delta and tokens saved, write Ablation docs. Components with delta <= noise floor, smallest first, become delete-edit candidates; unused tools are deleted without ablation.
5. hx/promote.py: gate and merge_promoted per the SPEC section 5 gate. merge_promoted applies all winning edits to the parent in one new harness with status promoted; losing candidates are marked rejected (keep them, the lineage tree shows them).
6. hx/analytics.py: prediction_scores: for each manifest, precision = |predicted_fixes ∩ actual_fixed| / |predicted_fixes|, recall = |predicted_fixes ∩ actual_fixed| / |actual_fixed|, and the same for regressions; aggregate per generation.
7. hx/loop.py and scripts/run_evolution.py: the loop in SPEC section 5, cadence policy, growth, growth, prune repeating, logging one line per generation (scores, tokens, n_rules, edits promoted/rejected). Resumable from a harness id. Support --start so we can run analytics from the evolved mflix harness (carried) and from h0 (cold).
8. Stretch: score the meta-agent's own prompt as a versioned artifact using prediction_scores.

Before trusting anything: when A and B land (about 2:00), hand-check 10 debugger reports against the traces. If the component attribution is wrong, fix the debugger prompt before running the full loop.
```
