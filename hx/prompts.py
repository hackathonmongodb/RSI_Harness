"""Context assembly: pinned base + general harness rules + case index (+ selected case bodies)."""
from __future__ import annotations

import platform
import shutil
import sys
from datetime import date
from pathlib import Path

from hx.memory.cases import render_case, render_index
from hx.memory.harness import render_rules
from hx.memory.models import Case, HarnessVersion

BASE = """You are hx, a coding agent running in the user's terminal. You work directly on their computer:
you can read, create and edit files and run shell commands in the working directory.

How to work:
- Look before you change: list/read the relevant files first. Read a file before editing or overwriting it.
- Prefer edit_file for small changes and write_file for new files or full rewrites.
- After changing code, verify it: run it, its tests, or at least a syntax/import check.
- When a command fails, read the error and fix the cause. Do not repeat a failing command unchanged.
- The shell is non-interactive (no TTY, no stdin, no display) and each call starts in the working directory.
- Keep replies short and concrete. The user already sees your tool calls; end with what you did and how to run it."""

MEMORY_NOTE = """Memory: "Harness rules" are lessons learned from your own past failures; follow them.
Rules marked (trial) are new and still being tested. The "Case library" lists past incidents by one line;
the relevant ones are expanded in the user's message. Call read_case(id) if another one clearly applies."""


def environment_block(workdir: Path) -> str:
    shell = Path(shutil.which("bash") or "/bin/bash").name
    git = (workdir / ".git").exists()
    return (
        f"Working directory: {workdir}\n"
        f"Git repo: {'yes' if git else 'no'}\n"
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()}), shell: {shell}, "
        f"python: {Path(sys.executable).name} {platform.python_version()} (system python3 may differ)\n"
        f"Today: {date.today().isoformat()}"
    )


def system_prompt(workdir: Path, harness: HarnessVersion | None, cases: list[Case], memory: bool = True) -> str:
    parts = [BASE, "# Environment\n" + environment_block(workdir)]
    if memory:
        parts.append(MEMORY_NOTE)
        rules = render_rules(harness) if harness else ""
        parts.append("# Harness rules\n" + (rules or "(none yet)"))
        index = render_index(cases)
        if index:
            parts.append("# Case library (index)\n" + index)
    return "\n\n".join(parts)


def user_message(task: str, selected: list[Case]) -> str:
    if not selected:
        return task
    cases = "\n\n".join(render_case(c) for c in selected)
    return (
        "<recalled-cases>\nPast incidents that look relevant to this request. Use them if they apply.\n\n"
        f"{cases}\n</recalled-cases>\n\n{task}"
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
