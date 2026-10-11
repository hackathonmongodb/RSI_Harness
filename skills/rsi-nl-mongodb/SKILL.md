---
name: rsi-nl-mongodb
description: All reads against the RSI harness database MUST go through MongoDB Natural Language Querying (schema + samples + indexes → mongosh). Use when querying harness_versions, eval_runs, memories, query_traces, or agent_events.
---

# RSI: Natural Language to MongoDB only

Do not hand-write `find` / `aggregate` in application agent loops.

1. Load `skills/mongodb-natural-language-querying/SKILL.md`.
2. Introspect indexes, annotated schema, and 2–3 sample documents.
3. Generate mongosh using the official MongoDB prompt template.
4. Execute read-only (`find` or `aggregate` without `$out`/`$merge`).
5. Record a `query_traces` document.

Writes to `harness_versions` happen only through the evolve path after an NL query of `eval_runs`.
