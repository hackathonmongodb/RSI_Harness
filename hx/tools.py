"""The agent's hands: read / write / edit files, list, grep, run shell, open a memory case.

Borrowed from Claude Code's basic tools, minus everything else: a file must be read before it
is overwritten or edited, edits are exact-string and must be unique, shell output is capped.
"""
from __future__ import annotations

import fnmatch
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".hx"}
READ_DEFAULT_LIMIT = 250
READ_MAX_CHARS = 12_000
BASH_MAX_CHARS = 5_000
LIST_MAX = 200
GREP_MAX = 100

Confirm = Callable[[str], bool]


@dataclass
class ToolResult:
    ok: bool
    output: str
    kind: str = "ok"   # ok | error | nonzero_exit | timeout | denied


@dataclass
class ToolContext:
    workdir: Path
    auto_approve: bool = False
    bash_timeout: int = 60
    confirm: Optional[Confirm] = None
    load_case: Optional[Callable[[str], Optional[str]]] = None
    read_state: dict[str, int] = field(default_factory=dict)   # abs path -> mtime_ns at last read/write

    def resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(str(path or ".")))
        if not p.is_absolute():
            p = self.workdir / p
        return p.resolve()

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.workdir.resolve()))
        except ValueError:
            return str(p)

    def inside(self, p: Path) -> bool:
        try:
            p.relative_to(self.workdir.resolve())
            return True
        except ValueError:
            return False

    def ask(self, prompt: str) -> bool:
        if self.auto_approve:
            return True
        if self.confirm is None:
            return False
        return bool(self.confirm(prompt))


# ---------------------------------------------------------------------------- helpers
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Za-z0-9]|\x1b[=>78]|\r(?!\n)")


def _strip_ansi(text: str) -> str:
    """Terminal control sequences are noise to the model (and break the UI)."""
    return _ANSI.sub("", text)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} chars clipped] ...\n{text[-tail:]}"


def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as fh:
            return b"\0" in fh.read(4096)
    except OSError:
        return False


def _mtime(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return -1


def _check_fresh(ctx: ToolContext, p: Path) -> Optional[str]:
    key = str(p)
    if key not in ctx.read_state:
        return f"{ctx.rel(p)} exists but was not read in this session. Call read_file on it first."
    if ctx.read_state[key] != _mtime(p):
        return f"{ctx.rel(p)} changed on disk since you read it. Call read_file again first."
    return None


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for f in sorted(filenames):
            yield Path(dirpath) / f


# ---------------------------------------------------------------------------- tools
def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = READ_DEFAULT_LIMIT) -> ToolResult:
    p = ctx.resolve(path)
    if not p.exists():
        return ToolResult(False, f"File not found: {ctx.rel(p)}", "error")
    if p.is_dir():
        return ToolResult(False, f"{ctx.rel(p)} is a directory; use list_files.", "error")
    if _is_binary(p):
        return ToolResult(False, f"{ctx.rel(p)} looks binary; not reading it.", "error")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    ctx.read_state[str(p)] = _mtime(p)
    if not lines:
        return ToolResult(True, f"({ctx.rel(p)} is empty)")
    start = max(1, int(offset or 1))
    lim = max(1, int(limit or READ_DEFAULT_LIMIT))
    chunk = lines[start - 1 : start - 1 + lim]
    out, size = [], 0
    for i, line in enumerate(chunk, start):
        row = f"{i:>5}\t{line[:500]}"
        size += len(row) + 1
        if size > READ_MAX_CHARS:
            out.append(f"... (stopped at {READ_MAX_CHARS} chars; continue with offset={i})")
            break
        out.append(row)
    shown_end = start + len(chunk) - 1
    if shown_end < len(lines) and not out[-1].startswith("..."):
        out.append(f"... ({len(lines) - shown_end} more lines; continue with offset={shown_end + 1})")
    return ToolResult(True, "\n".join(out))


def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    p = ctx.resolve(path)
    if p.is_dir():
        return ToolResult(False, f"{ctx.rel(p)} is a directory.", "error")
    existed = p.exists()
    if existed:
        stale = _check_fresh(ctx, p)
        if stale:
            return ToolResult(False, stale, "error")
    if not ctx.inside(p) and not ctx.ask(f"Write outside the working directory: {p}?"):
        return ToolResult(False, "User denied the write.", "denied")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = content if isinstance(content, str) else str(content)
    p.write_text(text, encoding="utf-8")
    ctx.read_state[str(p)] = _mtime(p)
    n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    return ToolResult(True, f"{'Updated' if existed else 'Created'} {ctx.rel(p)} ({n} lines)")


def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
    p = ctx.resolve(path)
    if not p.exists():
        return ToolResult(False, f"File not found: {ctx.rel(p)}. Use write_file to create it.", "error")
    stale = _check_fresh(ctx, p)
    if stale:
        return ToolResult(False, stale, "error")
    if old_string == new_string:
        return ToolResult(False, "old_string and new_string are identical.", "error")
    if not old_string:
        return ToolResult(False, "old_string is empty. Use write_file to replace a whole file.", "error")
    if not ctx.inside(p) and not ctx.ask(f"Edit outside the working directory: {p}?"):
        return ToolResult(False, "User denied the edit.", "denied")
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        hint = ""
        if old_string.strip() and old_string.strip() in text:
            hint = " (it matches if you ignore leading/trailing whitespace — copy indentation exactly)"
        return ToolResult(False, f"old_string not found in {ctx.rel(p)}{hint}. Re-read the file and copy the text exactly.", "error")
    if count > 1 and not replace_all:
        return ToolResult(False, f"old_string occurs {count} times in {ctx.rel(p)}. Add surrounding lines to make it unique, or set replace_all.", "error")
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    ctx.read_state[str(p)] = _mtime(p)
    at = new_text.find(new_string) if new_string else text.find(old_string)
    line_no = new_text.count("\n", 0, max(at, 0)) + 1
    lines = new_text.splitlines()
    lo, hi = max(1, line_no - 2), min(len(lines), line_no + new_string.count("\n") + 2)
    snippet = "\n".join(f"{i:>5}\t{lines[i - 1][:300]}" for i in range(lo, hi + 1))
    return ToolResult(True, f"Edited {ctx.rel(p)} ({count if replace_all else 1} replacement):\n{snippet}")


def list_files(ctx: ToolContext, path: str = ".", pattern: str = "*") -> ToolResult:
    root = ctx.resolve(path)
    if not root.exists():
        return ToolResult(False, f"Not found: {ctx.rel(root)}", "error")
    if root.is_file():
        return ToolResult(True, ctx.rel(root))
    pat = pattern or "*"
    out = []
    for f in _walk(root):
        rel = str(f.relative_to(root))
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat):
            out.append(ctx.rel(f))
            if len(out) >= LIST_MAX:
                out.append(f"... (stopped at {LIST_MAX} entries; narrow the pattern)")
                break
    return ToolResult(True, "\n".join(out) if out else "(no files)")


def grep(ctx: ToolContext, pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = False) -> ToolResult:
    try:
        rx = re.compile(pattern, re.I if ignore_case else 0)
    except re.error as err:
        return ToolResult(False, f"Bad regex: {err}", "error")
    root = ctx.resolve(path)
    files = [root] if root.is_file() else list(_walk(root))
    hits = []
    for f in files:
        if glob and glob != "*" and not fnmatch.fnmatch(f.name, glob) and not fnmatch.fnmatch(ctx.rel(f), glob):
            continue
        try:
            if f.stat().st_size > 1_000_000 or _is_binary(f):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{ctx.rel(f)}:{i}: {line.strip()[:300]}")
                    if len(hits) >= GREP_MAX:
                        hits.append(f"... (stopped at {GREP_MAX} matches)")
                        return ToolResult(True, "\n".join(hits))
        except OSError:
            continue
    return ToolResult(True, "\n".join(hits) if hits else "(no matches)")


def bash(ctx: ToolContext, command: str, timeout: Optional[int] = None) -> ToolResult:
    command = (command or "").strip()
    if not command:
        return ToolResult(False, "Empty command.", "error")
    if not ctx.ask(f"Run: {command}"):
        return ToolResult(False, "User denied this command.", "denied")
    limit = int(timeout or ctx.bash_timeout)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PAGER="cat", GIT_PAGER="cat")
    try:
        proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=str(ctx.workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
            start_new_session=True,   # so a timeout kills the whole process group
        )
    except OSError as err:
        return ToolResult(False, f"Could not start shell: {err}", "error")
    try:
        out, _ = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = proc.communicate()
        body = _clip(_strip_ansi(out or ""), BASH_MAX_CHARS)
        return ToolResult(
            False,
            f"Timed out after {limit}s and was killed. The command may be interactive (waits for a TTY/GUI/input) "
            f"or long-running.\n{body}",
            "timeout",
        )
    body = _clip(_strip_ansi(out or "").rstrip(), BASH_MAX_CHARS)
    rc = proc.returncode
    text = f"exit code: {rc}\n{body}" if body else f"exit code: {rc} (no output)"
    return ToolResult(rc == 0, text, "ok" if rc == 0 else "nonzero_exit")


def read_case(ctx: ToolContext, case_id: str) -> ToolResult:
    if ctx.load_case is None:
        return ToolResult(False, "Memory is disabled.", "error")
    body = ctx.load_case(str(case_id or "").strip())
    if body is None:
        return ToolResult(False, f"No case with id {case_id!r}.", "error")
    return ToolResult(True, body)


# ---------------------------------------------------------------------------- registry
def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


SCHEMAS: list[dict[str, Any]] = [
    _fn("read_file", "Read a text file. Returns numbered lines. Use offset/limit for long files.",
        {"path": {"type": "string"}, "offset": {"type": "integer", "description": "1-based start line"},
         "limit": {"type": "integer"}}, ["path"]),
    _fn("write_file", "Create a file or overwrite it entirely (read it first if it exists). Creates parent dirs.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace an exact, unique string in a file you have read. Copy indentation exactly.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
         "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"]),
    _fn("list_files", "List files under a directory (recursive, skips .git/.venv/node_modules). pattern is a glob like '*.py'.",
        {"path": {"type": "string"}, "pattern": {"type": "string"}}, []),
    _fn("grep", "Search file contents with a Python regex. Returns file:line: text.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
         "ignore_case": {"type": "boolean"}}, ["pattern"]),
    _fn("bash", "Run a shell command in the working directory (non-interactive: no TTY, stdin closed). "
                "Returns exit code and output. cwd does not persist between calls.",
        {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds"}}, ["command"]),
    _fn("read_case", "Open a case from the case library by id (listed in the system prompt).",
        {"case_id": {"type": "string"}}, ["case_id"]),
]

_IMPL: dict[str, Callable[..., ToolResult]] = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "grep": grep,
    "bash": bash,
    "read_case": read_case,
}


def schemas(memory: bool = True) -> list[dict[str, Any]]:
    return [s for s in SCHEMAS if memory or s["function"]["name"] != "read_case"]


def execute(ctx: ToolContext, name: str, args: dict[str, Any]) -> ToolResult:
    fn = _IMPL.get(name)
    if fn is None:
        return ToolResult(False, f"Unknown tool {name!r}. Available: {', '.join(_IMPL)}", "error")
    if "_unparsed" in args:
        return ToolResult(False, "Tool arguments were not valid JSON. Resend the call with a JSON object.", "error")
    try:
        return fn(ctx, **args)
    except TypeError as err:
        return ToolResult(False, f"Bad arguments for {name}: {err}", "error")
    except Exception as err:  # noqa: BLE001 — a tool crash is evidence, not a fatal error
        return ToolResult(False, f"{type(err).__name__}: {err}", "error")


_BENIGN_RC1 = re.compile(r"^\s*(grep|rg|egrep|fgrep|diff|cmp|test|\[|which|command -v|git diff|pgrep)\b")


def is_benign(name: str, args: dict[str, Any], result: ToolResult) -> bool:
    """Non-zero exits that are answers, not failures (grep found nothing, diff found a difference)."""
    if result.ok or result.kind == "denied":
        return True
    if name == "bash" and result.kind == "nonzero_exit":
        first = (args.get("command") or "").split("&&")[-1].split("|")[0]
        return result.output.startswith("exit code: 1") and bool(_BENIGN_RC1.match(first))
    return False


def summarize_args(name: str, args: dict[str, Any], limit: int = 160) -> str:
    """One-line description of a call for the UI and traces (never the full file content)."""
    if name in ("write_file",):
        body = args.get("content") or ""
        s = f"{args.get('path')} ({len(body)} chars)"
    elif name == "edit_file":
        s = f"{args.get('path')}"
    elif name == "bash":
        s = str(args.get("command", ""))
    elif name == "read_file":
        s = str(args.get("path"))
        if args.get("offset"):
            s += f":{args.get('offset')}"
    elif name == "grep":
        s = f"/{args.get('pattern')}/ in {args.get('path', '.')}"
    elif name == "list_files":
        s = f"{args.get('path', '.')} {args.get('pattern') or ''}".strip()
    else:
        s = ", ".join(f"{k}={v}" for k, v in args.items())
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"
