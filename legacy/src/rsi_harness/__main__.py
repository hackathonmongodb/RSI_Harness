from __future__ import annotations

import argparse
import json

from rsi_harness.evolve import evolve_once
from rsi_harness.nl_query import run as nl_run
from rsi_harness.seed import seed


def main() -> None:
    p = argparse.ArgumentParser(prog="rsi-harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="Load baseline harness versions + evals into Atlas")

    q = sub.add_parser("nl", help="Natural language → mongosh → execute (read-only)")
    q.add_argument("query")

    e = sub.add_parser("evolve", help="Query scores via NL-to-MongoDB, then write a new harness")
    e.add_argument("--focus", default=None)

    s = sub.add_parser("serve", help="Open the live demo")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args()
    if args.cmd == "seed":
        print(json.dumps(seed(), indent=2))
    elif args.cmd == "nl":
        print(json.dumps(nl_run(args.query), indent=2, default=str))
    elif args.cmd == "evolve":
        print(json.dumps(evolve_once(args.focus), indent=2, default=str))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "rsi_harness.server:app",
            host="127.0.0.1",
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
