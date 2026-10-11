# RSI Harness

Recursive self-improving **agent harness** for the Harness Engineering & Model Wrangling hackathon (Statement 1, with a long-horizon memory store).

**Every read against Atlas goes through MongoDB’s official Natural Language → mongosh path** — schema, sample documents, indexes, and the published prompt template — not hand-written queries in the agent loop.

Docs followed: [Natural Language to MongoDB Queries](https://www.mongodb.com/docs/manual/natural-language-to-mongodb/)  
Official skill vendored at `skills/mongodb-natural-language-querying/`.

## Why this is not a banned starter

The product is a **mutating harness** (rules, context policy, guardrails, tool access) stored as versioned documents. Eval scores are hard metrics. The agent inspects those scores via NL→MongoDB, then writes the next harness generation.

## Atlas collections

| Collection | Role |
|---|---|
| `harness_versions` | Full environment snapshot, lineage via `parent_version` |
| `eval_runs` | Fitness signal for RSI |
| `memories` | Long-horizon notes (user / project / team / agent) |
| `query_traces` | Audit of every NL → mongosh generation |
| `agent_events` | Append-only long-horizon log |

## Part B runtime (`hx/`)

See `docs/PART_B.md`. C imports `complete`, `assemble`, `apply_edits`, `run_task`, `run_generation`.

```bash
source .venv/bin/activate
export LLM_FAKE=1
pytest -q
python scripts/run_baseline.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env
# put the Atlas Sandbox URI + ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

```bash
PYTHONPATH=src python -m rsi_harness seed
PYTHONPATH=src python -m rsi_harness nl "Average eval score grouped by harness version"
PYTHONPATH=src python -m rsi_harness evolve
PYTHONPATH=src python -m rsi_harness serve
```

Open http://127.0.0.1:8000

## Demo script (~3 minutes)

1. Reseed sandbox → show lineage v1→v3 in MongoDB.
2. Type English → watch **mongosh** appear → results from Atlas.
3. Click **Evolve from scores** → the loop queries `eval_runs` through the same NL path, inserts `harness_versions` v4, promotes it to production.

Judges should see original work: NL query compiler, read-only execution, RSI promotion — not a dashboard and not generic RAG.
