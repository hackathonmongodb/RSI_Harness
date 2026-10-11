from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import json_util

from rsi_harness.db import db
from rsi_harness.models import COLLECTION_DOCS, EvalRun, HarnessVersion, Memory, QueryTrace

SCHEMA_BY_COLLECTION = {
    "harness_versions": HarnessVersion,
    "eval_runs": EvalRun,
    "memories": Memory,
    "query_traces": QueryTrace,
}


def _truncate(doc: dict[str, Any], str_limit: int = 120) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if isinstance(v, str) and len(v) > str_limit:
            out[k] = v[:str_limit] + "…"
        elif isinstance(v, list) and len(v) > 6:
            out[k] = v[:6] + ["…"]
        elif isinstance(v, dict):
            out[k] = _truncate(v, str_limit)
        else:
            out[k] = v
    return out


def indexes(collection: str) -> list[dict[str, Any]]:
    return list(db()[collection].list_indexes())


def sample_documents(collection: str, limit: int = 3) -> list[dict[str, Any]]:
    docs = list(db()[collection].find().limit(limit))
    return [_truncate(d) for d in docs]


def annotated_schema(collection: str) -> str:
    model = SCHEMA_BY_COLLECTION.get(collection)
    if model is None:
        return "{}"
    return model.model_json_schema().__str__()


def collection_prompt_block(collection: str) -> str:
    samples = json_util.dumps(sample_documents(collection), indent=2)
    idxs = json_util.dumps(indexes(collection), indent=2)
    desc = COLLECTION_DOCS.get(collection, "")
    return f"""#### Collection `{collection}`
Description: {desc}

Schema:
```
{annotated_schema(collection)}
```

Example documents:
```
{samples}
```

Indexes:
```
{idxs}
```
"""


def database_prompt(collections: list[str] | None = None) -> str:
    names = collections or list(COLLECTION_DOCS)
    blocks = "\n".join(collection_prompt_block(c) for c in names)
    latest = datetime.now().astimezone().isoformat()
    return f"""## Database Information

Name: rsi_harness
Description: Recursive self-improving agent harness store. Harness versions, eval scores, memories, and NL query traces live here.
Latest Date: {latest} (use this to inform dates in queries)

### Collections

{blocks}
"""
