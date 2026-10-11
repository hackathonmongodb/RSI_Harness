# SPEC.md — a harness that evolves itself, pointed at a MongoDB cluster

Contract for all three workstreams. `hx/models.py` is the source of truth for field names; this file defines semantics, permissions, signatures and the loop. Change either one and tell the team in the same commit.

## 1. Pitch
Every company's MongoDB schema is weird. Instead of hand-tuning prompts per database, this harness tunes itself to whatever cluster you point it at: it rewrites its own schema context, rules and retrieval settings from failure evidence, and deletes whatever stops earning its place. Task: natural language -> aggregation pipeline, scored by execution match against gold results. No LLM judge.

Headline result we want on screen: rule count and tokens-per-task go DOWN at each prune while held-out score holds or rises, and a harness evolved on `sample_mflix` adapts to `sample_analytics` in fewer generations than a cold start.

## 2. Databases, users, permissions (everything lives in the hackathon Atlas sandbox cluster)

| DB | Collections | Who may touch it |
|---|---|---|
| `sample_mflix`, `sample_analytics` (`sample_airbnb` if time) | Atlas sample data | role `task`: read. Used by the executor and the task agent's tools. |
| `hx_eval` | `tasks`, `gold`, `heldout_results`, `noise` | role `admin` only. Scorer lives in the orchestrator. |
| `hx_evidence` | `traces` (train split only), `reports` | role `meta`: read. Orchestrator (`admin`) writes. |
| `hx_harness` | `components`, `harnesses`, `manifests`, `ablations`, `dataset_context` | role `meta`: readWrite. |

Three DB users: `admin` (orchestrator), `task` (read on sample DBs), `meta` (read `hx_evidence`, readWrite `hx_harness`). Per-database built-in roles are enough, so this should work without custom roles. Demo beat: `scripts/demo_denied.py` makes the `meta` user try to edit a gold doc and a held-out result and prints the Unauthorized errors. If the sandbox tier refuses per-DB roles, enforce the same split in code and say so in the README.

`$jsonSchema` validators (A creates in `scripts/init_db.py`): `components` requires kind in {rule, policy, tool}; `content` max 400 chars; `value` only for policies; `status` enum. `manifests` requires non-empty `root_cause`, `targeted_fix`, `predicted_fixes` array present. Writes that don't validate fail at the database, which is the guardrail story.

## 3. Tasks and scoring
- Per dataset: about 30 train + 12 held-out tasks. Questions must name the exact output fields and any ordering / limit, and use absolute dates only.
- Spread across: single-stage filter, group/count, unwind + group, `$lookup`, date ranges, nested fields, array operators, Decimal128 / type quirks. Tag each task (`tags`).
- Candidate pipeline and gold pipeline both run read-only. Compare after `normalize`: Decimal128/Int64/Int32/float -> rounded float (4 dp), ObjectId -> str, datetime -> ISO, keys sorted. Unordered tasks compare as multisets; `ordered=True` compares sequences. Extra or missing fields = fail. Score is 0 or 1.
- `error_type`: `parse` (not valid JSON / EJSON), `blocked` (validator), `exec_error`, `timeout`, `wrong_result`.
- Gold is LLM-drafted then executed; drop anything that errors or returns empty; a human verifies at least 10 per dataset (`Gold.verified`).
- Seed noise: run the Gen-0 harness 3 times; store std in `hx_eval.noise`. `noise = 2 * std` is the promotion margin.

## 4. The harness

### Agent output contract
The task agent returns ONLY a JSON object: `{"collection": "movies", "pipeline": [ ... ]}` in Extended JSON (`{"$date": "2010-01-01T00:00:00Z"}`, `{"$oid": "..."}`). Aggregation only (`find` is expressible as a pipeline). Blocked stages: `$out`, `$merge`, `$function`, `$accumulator`, `$where`. Max 12 stages. `maxTimeMS` 5000. Stretch: an agent may answer `{"tool": name, "args": {...}}`; the runtime expands it to a pipeline before scoring.

### Components (all stored in `hx_harness.components`)
- **rule**: text injected into the system prompt. Max 400 chars. Examples: schema gotchas, a few-shot example (tag name `fewshot.*`), a reminder about types. Soft cap of 12 active rules, so adding past the cap requires a paired delete.
- **policy**: a typed knob. The meta-agent may only `modify` existing policies, within range. Never add keys.
- **tool** (stretch): `template = {"description", "params": {name: type}, "pipeline": [... "{{param}}" ...]}`. Substitution is textual into EJSON then parsed; no code execution.
- **pinned** components (just `rule.base` in Gen-0) cannot be edited or deleted by the meta-agent.

### Policies (Gen-0 defaults in bold)
| component_id | range |
|---|---|
| `policy.n_sample_docs` | **0**, 1, 2, 3 |
| `policy.doc_truncate_chars` | 100 to 800, **300** |
| `policy.schema_style` | **"names"** (field names only), "ts_annotated" |
| `policy.include_indexes` | **false**, true |
| `policy.cot` | **false**, true |
| `policy.max_retries` | **0**, 1, 2 (retry once feeds the executor error back to the model) |
| `policy.sample_docs_tool` | **false**, true (agent may call `sample_docs(collection, n)`) |

### Harness assembly (B)
System prompt = `rule.base` (role + output contract, pinned) + active rules sorted by id + CoT instruction if `policy.cot`. User message = database block in the shape MongoDB recommends: database name and description, per collection a description + schema (per `schema_style`) + `n_sample_docs` sample docs truncated to `doc_truncate_chars` + indexes if enabled, then the question. Record every component version that went into the prompt in `Trace.context_components`.

### Reference baselines (both `status="static"`, drawn as flat lines on the score chart)
- `h0`: Gen-0. `rule.base` and default policies only. This is the evolution starting point.
- `h_mongodb`: MongoDB's recommended prompt: 9 general tips as one rule each, CoT on, 3 sample docs, ts_annotated schema, indexes on, 3 few-shot rules. The thing we're trying to beat or match with fewer tokens.

## 5. Generation loop

```
H = h0
for gen in 1..N:
    phase = cadence[gen]            # policy, growth, growth, prune, policy, growth, growth, prune, ...
    base = run_generation(H, dataset)                       # cached if already measured
    reports = analyze(H, base.failed_train_trace_ids + sample(base.passed_train_trace_ids, 5))
    overview = build_overview(H)
    manifests = propose_edits(H, phase, overview)           # policy <= 2, growth <= 3, prune <= 3 deletes
    candidates = [apply_edits(H, [m.edit]) for m in manifests]   # one candidate per edit => clean attribution
    results = run_generation(c) for each candidate, concurrently
    winners = [m for m, r in zip(manifests, results) if gate(base, r, m.edit.op, noise)]
    fill actual_fixed / actual_regressed / verdict on every manifest
    H = merge_promoted(H, winners)                          # next iteration measures the merged harness
```

Phase meanings:
- **policy**: modify a knob. Cheap and deterministic.
- **growth**: add rules from failure reports (`missing_component=True`) or modify a rule the debugger implicated.
- **prune**: `prune_candidates` runs leave-one-out ablation on every active non-pinned rule/tool, only on train tasks where it appeared in `context_components`. Components with delta at or below the noise floor become delete edits (smallest delta first, up to 3). Unused tools are deleted without ablation.

### Promotion gate
Let `noise` come from `hx_eval.noise`. For add / modify: `child.heldout >= base.heldout - noise` AND `child.train >= base.train + noise` (a real fix, not a wobble). For delete: `child.heldout >= base.heldout - noise` AND `child.train >= base.train - noise` AND `child.tokens_per_task < base.tokens_per_task`. Record the reason string in `Manifest.gate_reason`.

### Evidence layers (cheapest first)
1. `build_overview`: aggregation over `reports`: counts by `implicated_component_id`, `implicated_kind`, `error_type`, `missing_component`; top failing tags. This is what the evolve agent reads first.
2. `reports`: per-trace debugger output with root cause and implicated component.
3. `traces`: raw, fetched by id only when a report is unclear.
The debugger runs on all failed train traces plus up to 5 passes (for credit assignment), small-model, parallel. Spot-check 10 reports by hand early: if the component attribution is wrong, every downstream edit is untargeted.

### Manifests
Each edit carries `root_cause`, `targeted_fix`, `evidence_trace_ids`, `predicted_fixes`, `predicted_regressions` (train task ids). After evaluation, fill `actual_fixed` / `actual_regressed`. `analytics.prediction_scores` computes precision and recall per edit and per generation. This is the second curve on the dashboard: does the meta-agent understand its own harness.

### Hard constraints on the meta-agent
It can only write `hx_harness` (components, manifests, harnesses). It never reads gold, held-out tasks or held-out outcomes. It cannot touch pinned components or policy keys outside the table. Each rule edit is validated by `$jsonSchema` at the database.

## 6. Dashboard (Streamlit, A owns, from about 2:00)
1. Score curves: train and held-out per generation with phase markers; flat lines for `h0` and `h_mongodb`.
2. Component count (rules, tools) and tokens per task per generation. This is the pruning chart.
3. Lineage tree of harnesses (`st.graphviz_chart`), nodes colored by status, edge labels = op + component. Lineage queried with `$graphLookup` on `parent_id`.
4. Manifest viewer: pick an edit, see evidence, root cause, predicted vs actual, gate reason.
5. Prediction precision / recall per generation.
6. Transfer chart: generations-to-threshold on `sample_analytics`, cold (from `h0`) vs carried (from the evolved mflix harness).
Use `$setWindowFields` for moving averages. Use `$facet` for the overview.

## 7. Modules and signatures (owners in brackets)

```python
# ---- A: Eval & Data ------------------------------------------------------
# hx/config.py
def get_client(role: Literal["admin", "task", "meta"]) -> MongoClient
# hx/executor.py
def validate_pipeline(pipeline: list[dict]) -> None          # raises ValueError
def run_pipeline(dataset: str, collection: str, pipeline: list[dict],
                 max_ms: int = 5000, row_cap: int = 500) -> list[dict]   # role "task"
# hx/scorer.py
def normalize(docs: list[dict]) -> list[str]                 # canonical JSON strings
def score(task: Task, parsed: dict | None) -> tuple[float, ErrorType, str | None]
def get_noise_floor(dataset: str) -> float
# hx/schema_probe.py
def get_dataset_context(dataset: str) -> dict
#   {"description": str, "collections": {name: {"description": str, "ts_schema": str,
#                                               "sample_docs": [ejson_str], "indexes": [str]}}}
def load_tasks(dataset: str, split: Split) -> list[Task]

# ---- B: Runtime ----------------------------------------------------------
# hx/llm.py
class LLMResult(BaseModel):
    text: str; prompt_tokens: int; completion_tokens: int; tool_calls: list[dict] = []
def complete(model: str, system: str, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> LLMResult
# hx/harness.py
def load_harness(harness_id: str) -> tuple[Harness, list[Component]]
def assemble(harness: Harness, comps: list[Component], task: Task, ctx: dict
             ) -> tuple[str, list[dict], list[dict]]         # system, messages, tools
def apply_edits(parent_id: str, edits: list[Edit], phase: Phase, generation: int) -> Harness
#   writes new component versions + child harness (status="candidate"), returns child
# hx/task_agent.py
def run_task(harness_id: str, task: Task, seed: int = 0) -> Trace
# hx/generation.py
def run_generation(harness_id: str, dataset: str, seeds: list[int] = [0], parallel: int = 8,
                   task_ids: list[str] | None = None) -> GenerationResult
#   idempotent + cached per (harness_id, seed, task set). Writes train traces to hx_evidence,
#   held-out scores to hx_eval.heldout_results, and updates the harness doc's scores/tokens.

# ---- C: Evolution --------------------------------------------------------
# hx/debugger.py
def analyze(harness_id: str, trace_ids: list[str]) -> list[Report]
# hx/overview.py
def build_overview(harness_id: str) -> dict
# hx/evolve.py
def propose_edits(harness_id: str, phase: Phase, overview: dict) -> list[Manifest]
# hx/ablate.py
def prune_candidates(harness_id: str, dataset: str) -> list[Ablation]
# hx/promote.py
def gate(parent: GenerationResult, child: GenerationResult, op: Op, noise: float) -> tuple[bool, str]
def merge_promoted(parent_id: str, winners: list[Manifest], generation: int, phase: Phase) -> Harness
# hx/analytics.py
def prediction_scores(generation: int) -> dict
# hx/loop.py
def evolve(dataset: str, start_harness_id: str, generations: int = 10) -> str
```

Scripts: A: `init_db.py`, `load_samples.py`, `make_tasks.py`, `noise_floor.py`, `demo_denied.py`. B: `seed_baseline.py`, `run_baseline.py`. C: `run_evolution.py --dataset sample_mflix --gens 10 [--start HARNESS_ID]`. Dashboard: `dashboard/app.py` (A).

## 8. Schedule (today, cutoff 5:00 PM)
| Time | Milestone |
|---|---|
| 12:30 | Repo + kit pushed, cluster up, 3 DB users, sample data loaded, `.env` shared, `llm.py` pushed. Everyone running their first plan-mode session. |
| 2:00 | **Integration checkpoint**: `scripts/run_baseline.py` prints train/held-out score for `h0` and `h_mongodb` on mflix. If not green by 2:15, cut scope (section 9). |
| 3:15 | Evolution loop runs end to end; the full mflix run (about 10 generations) starts NOW at the latest. |
| 3:45 | Code freeze except bug fixes. mflix done. Analytics cold and carried runs going in parallel. |
| 4:15 | Record the 1-minute demo, verify audio and video. Write README and project description. |
| 4:30 | Submit on Cerebral Valley (repo public, demo link works, all 3 teammates added). 30 min buffer. |

## 9. Cut order if behind
1. `sample_airbnb` (never start it unless everything else is done)
2. Vector clustering of failure reports
3. Tool synthesis (`kind="tool"`); rules + policies already tell the pruning story
4. Evolving the evolver (scoring the meta-prompt by prediction accuracy)
5. Last resort: skip the analytics transfer run and show it on a small subset.
Never cut: held-out gate, manifests with predictions, pruning, DB-level permission demo.

## 10. Verify early (I could not check these from here)
- Cluster tier and storage. The sample datasets together are a few hundred MB; on a small free tier that may crowd out traces.
- Whether the sandbox allows per-database users with built-in roles.
- Actual field shapes in `sample_analytics` (and `sample_airbnb` Decimal128 fields). Use the MongoDB MCP server to look before writing tasks.
- Whether the Hugging Face benchmark `mongodb-eai/natural-language-to-mongosh` covers these datasets with executable gold; if it does it saves task-writing time. Note it is mongosh code, so it would need converting to pipelines.
