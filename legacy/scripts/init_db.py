#!/usr/bin/env python3
"""Create hx_* databases, indexes, and $jsonSchema validators. SPEC §2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx.config import get_client

COMPONENT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["component_id", "version", "kind", "name", "status"],
        "properties": {
            "component_id": {"bsonType": "string"},
            "version": {"bsonType": ["int", "long"]},
            "kind": {"enum": ["rule", "policy", "tool"]},
            "name": {"bsonType": "string"},
            "content": {"bsonType": ["string", "null"], "maxLength": 400},
            "value": {},
            "template": {"bsonType": ["object", "null"]},
            "pinned": {"bsonType": "bool"},
            "dataset_scope": {"bsonType": "string"},
            "status": {"enum": ["active", "retired", "rejected"]},
            "token_cost": {"bsonType": ["int", "long", "double"]},
            "created_gen": {"bsonType": ["int", "long"]},
            "created_by_edit": {"bsonType": ["string", "null"]},
        },
        "additionalProperties": True,
    }
}

MANIFEST_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "edit_id",
            "generation",
            "phase",
            "parent_harness_id",
            "edit",
            "root_cause",
            "targeted_fix",
            "predicted_fixes",
        ],
        "properties": {
            "edit_id": {"bsonType": "string"},
            "root_cause": {"bsonType": "string", "minLength": 1},
            "targeted_fix": {"bsonType": "string", "minLength": 1},
            "predicted_fixes": {"bsonType": "array"},
        },
        "additionalProperties": True,
    }
}


def _create_or_update(db, name: str, validator: dict | None = None) -> None:
    names = set(db.list_collection_names())
    if name not in names:
        if validator:
            db.create_collection(name, validator=validator, validationLevel="strict")
        else:
            db.create_collection(name)
        print(f"created {db.name}.{name}")
    elif validator:
        db.command(
            {
                "collMod": name,
                "validator": validator,
                "validationLevel": "strict",
            }
        )
        print(f"updated validator on {db.name}.{name}")
    else:
        print(f"exists {db.name}.{name}")


def init() -> None:
    admin = get_client("admin")

    hx_eval = admin["hx_eval"]
    for name in ("tasks", "gold", "heldout_results", "noise"):
        _create_or_update(hx_eval, name)
    hx_eval.tasks.create_index([("dataset", 1), ("split", 1), ("task_id", 1)], unique=True)
    hx_eval.gold.create_index("task_id", unique=True)
    hx_eval.heldout_results.create_index(
        [("harness_id", 1), ("task_id", 1), ("seed", 1)], unique=True
    )
    hx_eval.noise.create_index("dataset", unique=True)

    hx_evidence = admin["hx_evidence"]
    for name in ("traces", "reports"):
        _create_or_update(hx_evidence, name)
    hx_evidence.traces.create_index([("harness_id", 1), ("task_id", 1)])
    hx_evidence.traces.create_index("trace_id", unique=True)
    hx_evidence.reports.create_index("trace_id")
    hx_evidence.reports.create_index("implicated_component_id")

    hx_harness = admin["hx_harness"]
    _create_or_update(hx_harness, "components", COMPONENT_VALIDATOR)
    _create_or_update(hx_harness, "manifests", MANIFEST_VALIDATOR)
    for name in ("harnesses", "ablations", "dataset_context"):
        _create_or_update(hx_harness, name)
    hx_harness.components.create_index([("component_id", 1), ("version", 1)], unique=True)
    hx_harness.harnesses.create_index("parent_id")
    hx_harness.harnesses.create_index("harness_id", unique=True)
    hx_harness.manifests.create_index("edit_id", unique=True)
    hx_harness.dataset_context.create_index("dataset", unique=True)
    print("init_db complete")


def test_invalid_component_rejected() -> None:
    """Prove $jsonSchema rejects a bad component."""
    db = get_client("admin")["hx_harness"]
    bad = {
        "component_id": "rule.too_long",
        "version": 1,
        "kind": "rule",
        "name": "too long",
        "content": "x" * 401,
        "status": "active",
    }
    try:
        db.components.insert_one(bad)
    except Exception as err:  # noqa: BLE001
        print("invalid component rejected:", type(err).__name__, str(err)[:160])
        return
    db.components.delete_one({"component_id": "rule.too_long", "version": 1})
    raise SystemExit("ERROR: invalid component was accepted — validator not enforced")


if __name__ == "__main__":
    init()
    test_invalid_component_rejected()
