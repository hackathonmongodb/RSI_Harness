"""Terminal rendering. `UI` is the silent base (tests, scripting); `ConsoleUI` draws with rich."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from hx.memory.models import Case, HarnessVersion
    from hx.tools import ToolResult


class UI:
    def status(self, text: str):
        return nullcontext()

    def event(self, kind: str, /, **data) -> None:
        """Structured loop event (recall, assemble, llm_*, tool_*, review_*, memory_write ...).
        The terminal UI renders through the methods below; the web UI streams these."""

    def waiting(self, seconds: float, reason: str) -> None: ...
    def context(self, hv: "HarnessVersion", n_cases: int, selected: list["Case"], sys_tokens: int, memory: bool) -> None: ...
    def thought(self, text: str) -> None: ...
    def final(self, text: str) -> None: ...
    def tool_start(self, name: str, summary: str) -> None: ...
    def tool_end(self, name: str, result: "ToolResult", benign: bool) -> None: ...
    def notice(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...
    def memory(self, analysis: str, action: str, actions: list[str]) -> None: ...

    def confirm(self, prompt: str) -> bool:
        return False


class ConsoleUI(UI):
    def __init__(self) -> None:
        from rich.console import Console

        self.console = Console(highlight=False)
        self._status = None
        self.always = False

    @contextmanager
    def status(self, text: str) -> Iterator[None]:
        with self.console.status(f"[dim]{text}…[/]", spinner="dots") as st:
            self._status = st
            try:
                yield
            finally:
                self._status = None

    def waiting(self, seconds: float, reason: str) -> None:
        msg = f"[yellow]{reason}: waiting {seconds:.0f}s…[/]"
        if self._status is not None:
            self._status.update(msg)
        else:
            self.console.print(msg)

    def context(self, hv, n_cases, selected, sys_tokens, memory) -> None:
        if not memory:
            self.console.print(f"[dim]memory off · system {sys_tokens} tok[/]")
            return
        live = hv.live_rules()
        trial = sum(1 for r in live if r.status == "trial")
        loaded = ", ".join(c.case_id for c in selected) or "none"
        self.console.print(
            f"[dim]harness v{hv.version} · {len(live)} rules"
            f"{f' ({trial} trial)' if trial else ''} · cases {len(selected)}/{n_cases} loaded: {loaded}"
            f" · system {sys_tokens} tok[/]"
        )

    def thought(self, text: str) -> None:
        self.console.print(f"[italic]{text.strip()}[/]")

    def final(self, text: str) -> None:
        from rich.markdown import Markdown

        if text.strip():
            self.console.print()
            self.console.print(Markdown(text.strip()))

    def tool_start(self, name: str, summary: str) -> None:
        from rich.markup import escape

        self.console.print(f"[bold cyan]●[/] [bold]{name}[/] [dim]{escape(summary)}[/]")

    def tool_end(self, name: str, result, benign: bool) -> None:
        from rich.markup import escape

        lines = (result.output or "").splitlines()
        if result.ok:
            if name in ("read_file", "list_files", "grep"):
                head = f"{len(lines)} lines"
            else:
                head = escape((lines[0] if lines else "done")[:200])
                if len(lines) > 1:
                    head += f"  [dim](+{len(lines) - 1} lines)[/]"
            self.console.print(f"  [dim]⎿[/] {head}")
            return
        color = "yellow" if benign else "red"
        shown = lines[:8]
        body = "\n    ".join(escape(line[:200]) for line in shown)
        more = f"\n    [dim]… {len(lines) - 8} more lines[/]" if len(lines) > 8 else ""
        self.console.print(f"  [{color}]⎿ {body}[/]{more}")

    def notice(self, text: str) -> None:
        self.console.print(f"[yellow]{text}[/]")

    def error(self, text: str) -> None:
        from rich.markup import escape

        self.console.print(f"[bold red]error:[/] {escape(text)}")

    def memory(self, analysis: str, action: str, actions: list[str]) -> None:
        from rich.markup import escape

        if action == "none" and not actions:
            self.console.print(f"[magenta]🧠 self-review:[/] [dim]{escape(analysis or 'nothing worth remembering')}[/]")
            return
        self.console.print(f"[magenta]🧠 self-review:[/] {escape(analysis)}")
        for a in actions:
            self.console.print(f"   [bold magenta]{escape(a)}[/]")

    def confirm(self, prompt: str) -> bool:
        from rich.markup import escape

        if self.always:
            return True
        try:
            ans = self.console.input(f"[yellow]? {escape(prompt)}[/] [dim](y = yes, a = always, N = no)[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans == "a":
            self.always = True
            return True
        return ans in {"y", "yes"}
