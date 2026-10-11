"""`hx` — terminal entry point.

  hx                       interactive session in the current directory
  hx -p "build snake"      one task, then exit
  hx rules | cases | case ID | history | traces | forget ID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from hx.agent import Agent
from hx.config import Settings, load_settings
from hx.llm import LLM, LLMError
from hx.memory import cases as case_ops
from hx.memory import harness as rule_ops
from hx.memory.store import Store, open_store
from hx.ui import ConsoleUI

HELP = """[bold]Commands[/]
  /rules            general harness rules (loaded every turn)
  /cases            case library index
  /case ID          show one case
  /fail NOTE        the last result is wrong — self-review it with your note
  /retry            re-run the last task with a fresh context (memory carries over)
  /reflect          force a self-review of the last turn
  /forget ID        retire a rule or a case
  /history          harness versions (lineage)
  /traces           recent turns
  /memory on|off    toggle recall + self-review for this session
  /clear            forget the conversation (not the memory)
  /exit             quit
End a line with \\ to continue on the next line."""


# ---------------------------------------------------------------------------- printing memory
def print_rules(ui: ConsoleUI, store: Store, show_retired: bool = False) -> None:
    from rich.table import Table

    hv = store.current_harness()
    rows = hv.rules if show_retired else hv.live_rules()
    if not rows:
        ui.console.print(f"[dim]harness v{hv.version}: no rules yet[/]")
        return
    t = Table(title=f"harness v{hv.version}", show_lines=False, title_justify="left")
    for col in ("id", "status", "helped", "strikes", "rule"):
        t.add_column(col, overflow="fold")
    for r in rows:
        color = {"trial": "yellow", "active": "green", "retired": "dim"}[r.status]
        t.add_row(r.rule_id, f"[{color}]{r.status}[/]", str(r.helped), str(r.strikes), r.text)
    ui.console.print(t)


def print_cases(ui: ConsoleUI, store: Store) -> None:
    from rich.table import Table

    cs = store.list_cases()
    if not cs:
        ui.console.print("[dim]case library is empty[/]")
        return
    t = Table(title=f"case library ({len(cs)})", title_justify="left")
    for col in ("id", "uses", "helped", "description"):
        t.add_column(col, overflow="fold")
    for c in cs:
        t.add_row(c.case_id, str(c.uses), str(c.helped), c.description)
    ui.console.print(t)


def print_case(ui: ConsoleUI, store: Store, case_id: str) -> None:
    from rich.markdown import Markdown

    c = store.get_case(case_id)
    if c is None:
        ui.error(f"no case {case_id!r}")
        return
    ui.console.print(f"[bold]{c.case_id}[/] — {c.title}  [dim]({c.status}, used {c.uses}x, helped {c.helped}x)[/]")
    ui.console.print(f"[dim]{c.description}[/]")
    ui.console.print(Markdown(c.body))


def print_history(ui: ConsoleUI, store: Store) -> None:
    for hv in store.harness_history(20):
        ui.console.print(
            f"v{hv.version:<3} [dim]{hv.created_at:%Y-%m-%d %H:%M}[/]  {len(hv.live_rules())} rules  {hv.change}"
        )


def print_traces(ui: ConsoleUI, store: Store) -> None:
    color = {"ok": "green", "recovered": "yellow", "failed": "red", "aborted": "red"}
    for t in store.recent_traces(12):
        fails = len(t.failures())
        mem = "; ".join(a for a in t.memory_actions if not a.startswith("analysis:")) or "-"
        ui.console.print(
            f"[dim]{t.created_at:%m-%d %H:%M}[/] [{color[t.outcome]}]{t.outcome:<9}[/] "
            f"v{t.harness_version} cases={len(t.loaded_case_ids)} fails={fails} sys={t.system_tokens}tok "
            f"{t.task[:50]!r}  [magenta]{mem}[/]"
        )


def forget(ui: ConsoleUI, store: Store, item_id: str) -> None:
    hv = store.current_harness()
    r = hv.get(item_id)
    if r is not None and r.status != "retired":
        ui.console.print(rule_ops.revise_rule(store, item_id, retire=True, why="retired by user"))
    elif store.get_case(item_id) is not None:
        ui.console.print(case_ops.update_case(store, item_id, retire=True))
    else:
        ui.error(f"no live rule or case {item_id!r}")


# ---------------------------------------------------------------------------- session
def banner(ui: ConsoleUI, s: Settings, store: Store) -> None:
    hv = store.current_harness()
    n_cases = len(store.list_cases())
    ui.console.print(
        f"[bold]hx[/] · self-improving coding agent\n"
        f"[dim]model {s.model} · memory {store.name}\n"
        f"cwd {s.workdir} · harness v{hv.version}, {len(hv.live_rules())} rules, {n_cases} cases · /help[/]"
    )


def _read_line(ui: ConsoleUI) -> Optional[str]:
    lines = []
    prompt = "[bold green]›[/] "
    while True:
        try:
            line = ui.console.input(prompt)
        except EOFError:
            return None
        if line.endswith("\\"):
            lines.append(line[:-1])
            prompt = "[dim]…[/] "
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def _setup_readline(home: Path) -> None:
    try:
        import atexit
        import readline

        hist = home / "history"
        home.mkdir(parents=True, exist_ok=True)
        if hist.exists():
            readline.read_history_file(str(hist))
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, str(hist))
    except Exception:  # noqa: BLE001 — history is a convenience
        pass


def _run(agent: Agent, ui: ConsoleUI, task: str) -> None:
    try:
        agent.run_turn(task)
    except KeyboardInterrupt:
        ui.notice("interrupted")
    except LLMError as err:
        ui.error(str(err))


def repl(agent: Agent, ui: ConsoleUI) -> None:
    store = agent.store
    last_task: Optional[str] = None
    _setup_readline(agent.settings.home)
    while True:
        try:
            line = _read_line(ui)
        except KeyboardInterrupt:
            ui.console.print("[dim](ctrl-d or /exit to quit)[/]")
            continue
        if line is None:
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("exit", "quit", "q"):
                break
            elif cmd == "help":
                ui.console.print(HELP)
            elif cmd == "rules":
                print_rules(ui, store, show_retired=arg == "all")
            elif cmd == "cases":
                print_cases(ui, store)
            elif cmd == "case" and arg:
                print_case(ui, store, arg)
            elif cmd == "history":
                print_history(ui, store)
            elif cmd == "traces":
                print_traces(ui, store)
            elif cmd == "forget" and arg:
                forget(ui, store, arg)
            elif cmd == "clear":
                agent.reset()
                ui.console.print("[dim]conversation cleared (memory kept)[/]")
            elif cmd == "memory" and arg in ("on", "off"):
                agent.set_memory(arg == "on")
                ui.console.print(f"[dim]memory {arg}[/]")
            elif cmd == "fail":
                agent.flag_failure(arg)
                ui.console.print("[dim]/retry to re-run the task with the updated memory[/]")
            elif cmd == "reflect":
                if agent.last_trace is None:
                    ui.notice("no previous turn")
                else:
                    agent.reflect(agent.last_trace, force=True)
            elif cmd == "retry":
                if not last_task:
                    ui.notice("nothing to retry yet")
                    continue
                agent.reset()
                ui.console.rule("[dim]retry with fresh context[/]")
                _run(agent, ui, last_task)
            else:
                ui.notice(f"unknown command /{cmd} — /help")
            continue
        last_task = line
        _run(agent, ui, line)


def run_ui(settings: Settings, store: Store, port: int, open_browser: bool) -> int:
    from hx.server import EventBus, WebUI, serve

    bus = EventBus()
    web = WebUI(bus, auto_approve=settings.auto_approve)
    try:
        llm = LLM(settings, on_wait=web.waiting)
    except LLMError as err:
        print(f"hx: {err}", file=sys.stderr)
        return 2
    # Approvals go through the browser so the toggle there is the single source of truth.
    agent = Agent(settings=settings.with_(auto_approve=False), llm=llm, store=store, ui=web)
    serve(agent, store, bus, web, settings, port=port, open_browser=open_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hx", description="Self-improving terminal coding agent.")
    p.add_argument("command", nargs="?", choices=["chat", "ui", "rules", "cases", "case", "history", "traces", "forget"], default="chat")
    p.add_argument("arg", nargs="?", help="id for `case` / `forget`")
    p.add_argument("-p", "--prompt", help="run one task non-interactively and exit")
    p.add_argument("-C", "--cwd", help="working directory (default: current)")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve shell commands and outside-cwd writes")
    p.add_argument("--model", help="main model id")
    p.add_argument("--no-memory", action="store_true", help="disable recall + self-review (baseline)")
    p.add_argument("--port", type=int, default=8765, help="port for `hx ui`")
    p.add_argument("--no-browser", action="store_true", help="`hx ui`: don't open a browser")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    if not workdir.is_dir():
        print(f"hx: not a directory: {workdir}", file=sys.stderr)
        return 2
    overrides = {"workdir": workdir, "model": args.model}
    if args.yes:
        overrides["auto_approve"] = True
    if args.no_memory:
        overrides["memory"] = False
    settings = load_settings(**overrides)
    ui = ConsoleUI()
    store = open_store(settings.mongodb_uri, settings.db_name, settings.home, warn=ui.notice)

    if args.command == "ui":
        return run_ui(settings, store, args.port, not args.no_browser)

    if args.command != "chat":
        {
            "rules": lambda: print_rules(ui, store, show_retired=True),
            "cases": lambda: print_cases(ui, store),
            "case": lambda: print_case(ui, store, args.arg or ""),
            "history": lambda: print_history(ui, store),
            "traces": lambda: print_traces(ui, store),
            "forget": lambda: forget(ui, store, args.arg or ""),
        }[args.command]()
        return 0

    try:
        llm = LLM(settings, on_wait=ui.waiting)
    except LLMError as err:
        ui.error(str(err))
        return 2
    agent = Agent(settings=settings, llm=llm, store=store, ui=ui)
    if args.prompt:
        trace = agent.run_turn(args.prompt)
        return 0 if trace.outcome in ("ok", "recovered") else 1
    banner(ui, settings, store)
    repl(agent, ui)
    ui.console.print(f"[dim]tokens this session: {llm.total_prompt_tokens} in / {llm.total_completion_tokens} out[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
