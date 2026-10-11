# AGENTS.md — rules for every Cursor agent in this repo

Read this file, SPEC.md and hx/models.py before touching code. If Cursor doesn't pick up AGENTS.md on its own, copy it to `.cursor/rules/hackathon.mdc`.

## What we're building
A query agent (natural language -> MongoDB aggregation pipeline) wrapped in a self-improving harness. The harness is a set of versioned components (rules, policies, tools) stored in Atlas. A meta-agent reads failure evidence, edits components (add / modify / delete), and every edit carries a falsifiable prediction. A candidate is promoted only if it holds on a held-out split. The design borrows the three observability ideas from Agentic Harness Engineering (Lin et al. 2026, arXiv 2604.25850): explicit components, layered evidence, falsifiable edits. Cite it in the README.

## Stack
Python 3.11, pymongo, pydantic v2, Streamlit for the dashboard. All LLM calls go through `hx/llm.py`; model names come from env (`TASK_MODEL` small/cheap, `META_MODEL` stronger).

## Hard rules
1. SPEC.md is the contract. Do not change collection schemas, function signatures or file ownership without editing SPEC.md / hx/models.py in the same commit and telling the team.
2. Only edit files you own (SPEC.md section 7 lists owners). If you need a change in someone else's file, append a line to `docs/REQUESTS.md` instead.
3. Secrets live in env vars only. `.env` is gitignored, `.env.example` is committed. The repo is public: never commit a connection string.
4. Get a Mongo client only through `hx.config.get_client(role)` with role in {admin, task, meta}. Never use the admin client inside code that holds an LLM-driven agent.
5. Never execute model output as code. The task agent returns JSON `{"collection": ..., "pipeline": [...]}` in Extended JSON; it is parsed with `bson.json_util.loads`, checked by `executor.validate_pipeline`, and run with `maxTimeMS`. No `eval`, no mongosh subprocess.
6. Determinism: temperature 0 everywhere except where SPEC says otherwise. Record seeds. Noise will otherwise look like progress.
7. A full generation must finish in about 2 minutes. Use `ThreadPoolExecutor(max_workers=8)` for LLM calls and watch rate limits.
8. The evolve agent must never see held-out tasks, held-out outcomes or gold pipelines. Only train-split evidence goes into any meta-agent prompt.
9. Git: one branch per person (`a/eval`, `b/runtime`, `c/evolve`), small commits, rebase on main before pushing, merge to main about every 45 minutes. No force pushes.
10. Before inventing a helper, grep for it. Before changing behavior, run the tests for the module.
11. Write a unit test for every non-trivial function in your area. Tests that need Atlas must be marked `@pytest.mark.atlas`.

## Working style for Cursor sessions
- Start each session in plan mode. Produce the plan, flag anything ambiguous in SPEC.md, wait for approval, then implement.
- Use parallel subagents only for independent files (a test file, a script). Never have two agents editing the same file.
- Use the MongoDB MCP server (pointed at our sandbox cluster) to inspect real documents and run sample pipelines instead of guessing field names. Install MongoDB Agent Skills before writing queries.
- For MongoDB documentation, append `.md` to any docs URL to get markdown, or start from https://www.mongodb.com/docs/llms.txt.

## Reference: what MongoDB says about NL -> MongoDB queries
Source: https://www.mongodb.com/docs/manual/natural-language-to-mongodb/ (guidance is based on MongoDB's public benchmark, `mongodb-eai/natural-language-to-mongosh` on Hugging Face).

Inputs that improve query quality, in the page's own framing:
- **Base prompt**: state the task, and the exact output format. The page's example asks for mongosh code only, with no explanation or fences. We deliberately differ: our agent returns JSON in Extended JSON so we never execute JavaScript.
- **Annotated schema**: represent each collection with language-native types (TypeScript interface, Pydantic model, Go struct) and a comment on every field.
- **Sample documents**: 2 to 3 representative docs per collection, serialized as EJSON; truncate long strings, deep nesting and large arrays (keep a few elements of embeddings).
- **Index info**: lets the model write queries that use indexes.
- **Latest date**: models don't know today's date; tasks that use relative dates need it. Our gold tasks use absolute dates, so this matters little here.
- **Few-shot**: 1 to 3 (question, query) pairs similar to the target question.
- **Chain of thought**: asking the model to reason step by step before answering improves quality but costs tokens and latency. This is a natural policy knob.
- **General tips the page lists** (we do NOT hand-seed these into the evolving harness; the meta-agent should discover what it needs from failures, and they only appear in the static "MongoDB-recommended" comparison baseline):
  1. use proper operators and types (ObjectId, ISODate)
  2. use an aggregation pipeline for complex queries ($match, $group, $lookup)
  3. prefer indexed access, avoid $where and collection scans
  4. add $sort and $limit when sensible
  5. handle null / missing / empty arrays explicitly with $exists and $type
  6. never emit `_id: null` in result objects
  7. never use `new Date()` with no argument; always give an explicit date
  8. for Decimal128, prefer range queries over exact equality
  9. use $elemMatch, $all, $size for array matching
- **Model choice**: the page says models that do well on general benchmarks do well at query generation, so don't over-invest in picking.
