from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rsi_harness.db import db, ensure_indexes, ping


def seed(reset: bool = True) -> dict:
    ping()
    ensure_indexes()
    d = db()
    if reset:
        for name in (
            "harness_versions",
            "eval_runs",
            "memories",
            "query_traces",
            "agent_events",
        ):
            d[name].delete_many({})

    now = datetime.now(timezone.utc)
    harnesses = [
        {
            "version": 1,
            "status": "retired",
            "parent_version": None,
            "rules": [
                "Answer as fast as possible.",
                "Prefer short replies over citations.",
            ],
            "context_policy": "Load only the latest user message. Ignore prior evals.",
            "guardrails": ["Do not execute shell commands."],
            "tool_access": {
                "allowed": ["nl_mongodb", "read"],
                "denied": ["bash", "write"],
                "require_approval": [],
            },
            "mutation_reason": "initial baseline",
            "avg_score": 0.41,
            "created_at": now - timedelta(hours=5),
        },
        {
            "version": 2,
            "status": "retired",
            "parent_version": 1,
            "rules": [
                "Plan before editing.",
                "Cite files you change.",
                "Stop after 12 tool calls unless the score is still rising.",
            ],
            "context_policy": "Load project memories tagged 'conventions' plus last 3 eval notes.",
            "guardrails": [
                "Do not execute shell commands.",
                "Never mutate harness_versions except via the evolve path.",
            ],
            "tool_access": {
                "allowed": ["nl_mongodb", "read", "write", "memory_write"],
                "denied": ["bash"],
                "require_approval": ["write"],
            },
            "mutation_reason": "v1 skipped planning; evals failed on multi-file tasks",
            "avg_score": 0.58,
            "created_at": now - timedelta(hours=3),
        },
        {
            "version": 3,
            "status": "production",
            "parent_version": 2,
            "rules": [
                "Plan before editing.",
                "Query eval_runs via natural language to MongoDB before changing policy.",
                "Keep a running goal until hard metrics move.",
            ],
            "context_policy": "Load production harness, team memories, and last 8 eval_runs.",
            "guardrails": [
                "NL datastore access is read-only.",
                "Never mutate harness_versions except via the evolve path.",
                "Require approval for bash.",
            ],
            "tool_access": {
                "allowed": ["nl_mongodb", "read", "write", "memory_write", "bash"],
                "denied": [],
                "require_approval": ["bash"],
            },
            "mutation_reason": "v2 still had no metric loop; RSI requires querying scores first",
            "avg_score": 0.71,
            "created_at": now - timedelta(hours=1),
        },
    ]
    d.harness_versions.insert_many(harnesses)

    tasks = [
        (1, "multi-file refactor", 0.32, 4, "skipped planning"),
        (1, "long-horizon bug hunt", 0.28, 6, "lost the original goal"),
        (1, "write tests", 0.62, 1, "ok on small task"),
        (2, "multi-file refactor", 0.55, 2, "planned but no metric check"),
        (2, "long-horizon bug hunt", 0.49, 3, "context policy too thin"),
        (2, "write tests", 0.70, 0, "improved"),
        (3, "multi-file refactor", 0.74, 0, "queried past failures first"),
        (3, "long-horizon bug hunt", 0.68, 1, "kept goal in memories"),
        (3, "write tests", 0.71, 0, "stable"),
    ]
    evals = []
    for i, (ver, task, score, viol, notes) in enumerate(tasks):
        evals.append(
            {
                "harness_version": ver,
                "task": task,
                "score": score,
                "latency_ms": 1200 + i * 80,
                "tokens_used": 8000 + i * 400,
                "guardrail_violations": viol,
                "notes": notes,
                "created_at": now - timedelta(minutes=40 - i * 3),
            }
        )
    d.eval_runs.insert_many(evals)

    d.memories.insert_many(
        [
            {
                "tier": "project",
                "content": "Finalists must use the Atlas Sandbox cluster.",
                "tags": ["hackathon", "constraints"],
                "created_at": now - timedelta(hours=6),
            },
            {
                "tier": "agent",
                "content": "Low scores on long-horizon tasks correlate with dropping the original goal.",
                "tags": ["long-horizon", "rsi"],
                "created_at": now - timedelta(hours=2),
            },
            {
                "tier": "team",
                "content": "Demo must show mongosh generated from English, then a harness version bump.",
                "tags": ["demo"],
                "created_at": now - timedelta(minutes=50),
            },
        ]
    )

    d.agent_events.insert_one(
        {
            "type": "seed",
            "message": "Baseline harness v1–v3 loaded",
            "created_at": now,
        }
    )
    return {
        "harness_versions": d.harness_versions.count_documents({}),
        "eval_runs": d.eval_runs.count_documents({}),
        "memories": d.memories.count_documents({}),
        "db": d.name,
    }


if __name__ == "__main__":
    print(seed())
