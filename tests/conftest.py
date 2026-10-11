from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from hx.config import Settings
from hx.llm import Reply, ToolCall
from hx.memory.store import LocalStore


class FakeLLM:
    """Scripted stand-in for hx.llm.LLM.

    chat_script: list of Reply (or callables(messages) -> Reply), consumed in order.
    json_handler: (system, user) -> dict, for recall and self-review side calls.
    """

    def __init__(self, chat_script=None, json_handler: Callable[[str, str], dict] | None = None):
        self.chat_script = list(chat_script or [])
        self.json_handler = json_handler or (lambda s, u: {})
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.json_calls: list[tuple[str, str]] = []

    def chat(self, messages, tools=None, **kw) -> Reply:
        self.chat_calls.append([dict(m) for m in messages])
        if not self.chat_script:
            return Reply(content="done")
        item = self.chat_script.pop(0)
        return item(messages) if callable(item) else item

    def json(self, system, user, **kw) -> dict:
        self.json_calls.append((system, user))
        return self.json_handler(system, user)


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"c_{name}_{len(json.dumps(args))}", name=name, arguments=args, raw_arguments=json.dumps(args))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "home")


@pytest.fixture
def settings(workdir: Path, tmp_path: Path) -> Settings:
    return Settings(api_key="test", workdir=workdir, home=tmp_path / "home", auto_approve=True, max_steps=8)
