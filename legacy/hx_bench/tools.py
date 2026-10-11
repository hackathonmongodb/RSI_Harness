"""Tool catalog as data. Executors (A) bind names later; B only advertises schemas."""

from __future__ import annotations

from hx.models import ToolSpec

CATALOG: dict[str, ToolSpec] = {
    "nl_mongodb": ToolSpec(
        name="nl_mongodb",
        description=(
            "Read Atlas by natural language. The runtime turns English/Chinese into "
            "a read-only mongosh find/aggregate using schema + samples + indexes."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        read_only=True,
    ),
    "read": ToolSpec(
        name="read",
        description="Read a local file path.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        read_only=True,
    ),
    "write": ToolSpec(
        name="write",
        description="Write text to a local file path.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "contents": {"type": "string"},
            },
            "required": ["path", "contents"],
        },
    ),
    "bash": ToolSpec(
        name="bash",
        description="Run a shell command. Honor require_approval on the harness.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
    "memory_write": ToolSpec(
        name="memory_write",
        description="Persist a short note for later turns (long-horizon memory).",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tier": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    ),
}


def openai_tool(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }
