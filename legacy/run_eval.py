"""Run Stream A eval: harness answers → LLM-as-judge → results/<timestamp>.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_tasks
from llm import complete
from scoring import score

log = logging.getLogger("run_eval")

DEFAULT_HARNESS_SYSTEM = (
    "You are the RSI harness task agent for MongoDB sample_mflix. "
    "Answer the user's data question clearly and concretely. "
    "If you are unsure of exact counts, say so and give best effort."
)


def _harness_system() -> str:
    try:
        from rsi_harness.harness import active

        h = active()
        rules = h.get("rules") or []
        if rules:
            return DEFAULT_HARNESS_SYSTEM + "\nRules:\n- " + "\n- ".join(map(str, rules[:12]))
    except Exception:
        pass
    return DEFAULT_HARNESS_SYSTEM


def run_harness(prompt: str) -> tuple[str, dict]:
    text = complete(
        [
            {"role": "system", "content": _harness_system()},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    usage = dict(getattr(complete, "last_usage", {}) or {})
    return text, usage


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Groq Llama eval for RSI harness (Stream A)")
    p.add_argument("--tasks", default=None, help="Path to tasks.jsonl")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    tasks = load_tasks(args.tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}.jsonl"

    totals = {
        "n": 0,
        "scored": 0,
        "passes": 0,
        "score_sum": 0.0,
        "tokens": 0,
        "latency_ms": 0,
    }
    t0 = time.perf_counter()
    rows: list[dict] = []

    for task in tasks:
        totals["n"] += 1
        agent_out, harness_usage = run_harness(task["prompt"])
        totals["tokens"] += int(harness_usage.get("total_tokens") or 0)
        totals["latency_ms"] += int(harness_usage.get("latency_ms") or 0)

        verdict = score(task, agent_out, task.get("reference"))
        judge_usage = dict(getattr(complete, "last_usage", {}) or {})
        totals["tokens"] += int(judge_usage.get("total_tokens") or 0)
        totals["latency_ms"] += int(judge_usage.get("latency_ms") or 0)

        if verdict.get("score") is not None:
            totals["scored"] += 1
            totals["score_sum"] += float(verdict["score"])
            if verdict.get("pass"):
                totals["passes"] += 1

        row = {
            "task_id": task["id"],
            "prompt": task["prompt"],
            "reference": task.get("reference"),
            "agent_output": agent_out,
            "verdict": verdict,
            "harness_usage": harness_usage,
            "judge_usage": judge_usage,
        }
        rows.append(row)
        print(
            f"{task['id']}: score={verdict.get('score')} pass={verdict.get('pass')} "
            f"— {(verdict.get('reasoning') or '')[:80]}"
        )

    wall_s = time.perf_counter() - t0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean = (totals["score_sum"] / totals["scored"]) if totals["scored"] else None
    pass_rate = (totals["passes"] / totals["n"]) if totals["n"] else 0.0
    print()
    print("=== summary ===")
    print(f"tasks:        {totals['n']}")
    print(f"mean score:   {mean:.3f}" if mean is not None else "mean score:   n/a")
    print(f"pass rate:    {pass_rate:.1%} ({totals['passes']}/{totals['n']})")
    print(f"total tokens: {totals['tokens']}")
    print(f"total time:   {wall_s:.1f}s (api latency sum {totals['latency_ms']}ms)")
    print(f"wrote:        {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
