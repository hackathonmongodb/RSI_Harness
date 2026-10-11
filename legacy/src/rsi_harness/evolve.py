"""Recursive harness mutation: inspect Atlas via NL-to-MongoDB, then write a new version."""

from __future__ import annotations

import json
from typing import Any

from rsi_harness.harness import active, insert_version, promote
from rsi_harness.llm import complete, parse_json_object
from rsi_harness.nl_query import run as nl_run


EVOLVE_SYSTEM = """You improve an agent harness (rules, context policy, guardrails, tool access).
You receive: the current production harness, plus MongoDB query results about eval scores.
Propose the NEXT harness. Change at least one rule, guardrail, or tool policy based on the metrics.
Return ONLY JSON:
{
  "mutation_reason": "why, citing the metric evidence",
  "rules": ["..."],
  "context_policy": "...",
  "guardrails": ["..."],
  "tool_access": {
    "allowed": ["nl_mongodb", "memory_write", "bash", "read", "write"],
    "denied": [],
    "require_approval": ["bash"]
  }
}
Do not copy the parent verbatim. Keep lists short (3-8 items).
"""


def evolve_once(focus: str | None = None) -> dict[str, Any]:
    current = active()
    nl = focus or (
        "Average eval score and total guardrail violations grouped by harness_version, "
        "plus the 8 lowest scoring eval_runs with task and notes"
    )
    evidence = nl_run(nl, collections=["eval_runs", "harness_versions"])
    snapshot = {
        k: current.get(k)
        for k in (
            "version",
            "rules",
            "context_policy",
            "guardrails",
            "tool_access",
            "avg_score",
        )
    }
    user = f"""Current production harness:
{json.dumps(snapshot, default=str, indent=2)}

NL→MongoDB mongosh used:
{evidence.get('mongosh')}

Evidence documents:
{json.dumps(evidence.get('results'), default=str, indent=2)}
"""
    proposal = parse_json_object(complete(EVOLVE_SYSTEM, user))
    doc = insert_version(
        {
            "status": "candidate",
            "parent_version": current["version"],
            "rules": proposal.get("rules") or current.get("rules"),
            "context_policy": proposal.get("context_policy") or current.get("context_policy"),
            "guardrails": proposal.get("guardrails") or current.get("guardrails"),
            "tool_access": proposal.get("tool_access") or current.get("tool_access"),
            "mutation_reason": proposal.get("mutation_reason", ""),
        }
    )
    promote(doc["version"])
    return {
        "parent": current["version"],
        "new_version": doc["version"],
        "mutation_reason": doc.get("mutation_reason"),
        "evidence_mongosh": evidence.get("mongosh"),
        "evidence_nl": evidence.get("natural_language"),
        "harness": {
            "rules": doc.get("rules"),
            "context_policy": doc.get("context_policy"),
            "guardrails": doc.get("guardrails"),
            "tool_access": doc.get("tool_access"),
        },
    }
