"""`hx ui` — local web app: chat + live view of the agent loop + the memory it writes.

Stdlib only (ThreadingHTTPServer + Server-Sent Events). Binds to 127.0.0.1 and requires a per-launch
token on every API call, because this agent can run shell commands on the machine.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from hx.agent import Agent
from hx.config import Settings
from hx.memory.cases import case_markdown
from hx.memory.harness import harness_markdown
from hx.memory.store import Store
from hx.ui import UI

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_EVENTS = 5000


# ---------------------------------------------------------------------------- event bus
class EventBus:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._next = 1
        self._cond = threading.Condition()

    def emit(self, kind: str, /, **data: Any) -> dict[str, Any]:
        with self._cond:
            evt = {"id": self._next, "t": time.time(), "kind": kind, "data": data}
            self._next += 1
            self._events.append(evt)
            if len(self._events) > MAX_EVENTS:
                del self._events[: len(self._events) - MAX_EVENTS]
            self._cond.notify_all()
            return evt

    def since(self, last_id: int, timeout: float) -> list[dict[str, Any]]:
        with self._cond:
            if not self._events or self._events[-1]["id"] <= last_id:
                self._cond.wait(timeout)
            return [e for e in self._events if e["id"] > last_id]


# ---------------------------------------------------------------------------- UI adapter
class WebUI(UI):
    """Turns agent callbacks into bus events; approvals wait for the browser."""

    def __init__(self, bus: EventBus, auto_approve: bool = False) -> None:
        self.bus = bus
        self.auto_approve = auto_approve
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def event(self, kind: str, /, **data) -> None:
        self.bus.emit(kind, **data)

    def waiting(self, seconds: float, reason: str) -> None:
        self.bus.emit("waiting", seconds=round(seconds, 1), reason=reason)

    def thought(self, text: str) -> None:
        self.bus.emit("thought", text=text)

    def final(self, text: str) -> None:
        self.bus.emit("final", text=text)

    def notice(self, text: str) -> None:
        self.bus.emit("notice", text=text)

    def error(self, text: str) -> None:
        self.bus.emit("error", text=text)

    def confirm(self, prompt: str) -> bool:
        if self.auto_approve:
            self.bus.emit("confirm_auto", prompt=prompt)
            return True
        cid = secrets.token_hex(4)
        slot = {"event": threading.Event(), "answer": False}
        with self._lock:
            self._pending[cid] = slot
        self.bus.emit("confirm_request", id=cid, prompt=prompt)
        slot["event"].wait(600)
        with self._lock:
            self._pending.pop(cid, None)
        self.bus.emit("confirm_result", id=cid, approved=slot["answer"])
        return bool(slot["answer"])

    def answer(self, cid: str, answer: str) -> bool:
        with self._lock:
            slot = self._pending.get(cid)
        if slot is None:
            return False
        if answer == "always":
            self.auto_approve = True
        slot["answer"] = answer in ("yes", "always")
        slot["event"].set()
        return True

    def deny_all(self) -> None:
        with self._lock:
            slots = list(self._pending.values())
        for slot in slots:
            slot["answer"] = False
            slot["event"].set()


# ---------------------------------------------------------------------------- app state
class App:
    def __init__(self, agent: Agent, store: Store, bus: EventBus, ui: WebUI, settings: Settings) -> None:
        self.agent, self.store, self.bus, self.ui, self.settings = agent, store, bus, ui, settings
        self.token = secrets.token_urlsafe(18)
        self.busy = False
        self.last_task: Optional[str] = None
        self._lock = threading.Lock()

    def run_job(self, label: str, fn) -> tuple[bool, str]:
        with self._lock:
            if self.busy:
                return False, "agent is busy"
            self.busy = True

        def work():
            try:
                fn()
            except Exception as err:  # noqa: BLE001 — surface, don't kill the server
                self.bus.emit("error", text=f"{type(err).__name__}: {err}")
            finally:
                self.busy = False
                self.bus.emit("idle", job=label)

        threading.Thread(target=work, name=f"hx-{label}", daemon=True).start()
        return True, "started"

    # -- views
    def state(self) -> dict[str, Any]:
        hv = self.store.current_harness()
        return {
            "model": self.settings.model,
            "fast_model": self.settings.fast_model,
            "reflect_model": self.settings.reflect_model,
            "store": self.store.name,
            "workdir": str(self.agent.ctx.workdir),
            "harness_version": hv.version,
            "busy": self.busy,
            "memory": self.agent.settings.memory,
            "auto_approve": self.ui.auto_approve,
            "context_budget": self.settings.context_budget,
            "last_task": self.last_task,
            "last_event_id": self.bus._next - 1,
        }

    def memory(self) -> dict[str, Any]:
        hv = self.store.current_harness()
        cases = self.store.list_cases(include_retired=True)
        return {
            "harness": json.loads(hv.model_dump_json()),
            "harness_md": harness_markdown(hv),
            "history": [
                {"version": h.version, "parent": h.parent, "change": h.change, "reason": h.reason,
                 "created_at": h.created_at.isoformat(), "n_rules": len(h.live_rules()), "trace_id": h.trace_id}
                for h in self.store.harness_history(50)
            ],
            "cases": [
                {**json.loads(c.model_dump_json()), "markdown": case_markdown(c), "file": f"cases/{c.case_id}.md"}
                for c in cases
            ],
        }

    def traces(self) -> list[dict[str, Any]]:
        out = []
        for t in self.store.recent_traces(40):
            out.append({
                "trace_id": t.trace_id, "created_at": t.created_at.isoformat(), "task": t.task[:200],
                "outcome": t.outcome, "abort_reason": t.abort_reason, "steps": t.steps,
                "failures": len(t.failures()), "harness_version": t.harness_version,
                "loaded_case_ids": t.loaded_case_ids, "system_tokens": t.system_tokens,
                "tokens": t.prompt_tokens + t.completion_tokens, "memory_actions": t.memory_actions,
            })
        return out

    # -- actions
    def chat(self, text: str) -> tuple[bool, str]:
        text = text.strip()
        if not text:
            return False, "empty message"

        def job():
            self.last_task = text
            self.bus.emit("user_message", text=text)
            self.agent.run_turn(text)

        return self.run_job("chat", job)

    def command(self, cmd: str, arg: str) -> tuple[bool, str]:
        a = self.agent
        if cmd == "stop":
            a.cancel.set()
            self.ui.deny_all()
            self.bus.emit("notice", text="stop requested")
            return True, "stopping"
        if cmd == "auto_approve":
            self.ui.auto_approve = arg == "on"
            self.bus.emit("settings", auto_approve=self.ui.auto_approve)
            return True, "ok"
        if cmd == "memory":
            a.set_memory(arg == "on")
            self.bus.emit("settings", memory=a.settings.memory)
            return True, "ok"
        if cmd == "clear":
            if self.busy:
                return False, "agent is busy"
            a.reset()
            self.bus.emit("cleared")
            return True, "ok"
        if cmd == "retry":
            if not self.last_task:
                return False, "nothing to retry yet"
            task = self.last_task

            def job():
                a.reset()
                self.bus.emit("cleared", reason="retry with fresh context; memory carries over")
                self.bus.emit("user_message", text=task, retry=True)
                a.run_turn(task)

            return self.run_job("retry", job)
        if cmd == "fail":
            if a.last_trace is None:
                return False, "no previous turn"
            note = arg or "the result did not work"

            def job():
                self.bus.emit("user_feedback", text=note)
                a.flag_failure(note)

            return self.run_job("fail", job)
        if cmd == "reflect":
            if a.last_trace is None:
                return False, "no previous turn"
            trace = a.last_trace
            return self.run_job("reflect", lambda: a.reflect(trace, force=True))
        if cmd == "forget":
            hv = self.store.current_harness()
            r = hv.get(arg)
            from hx.memory import cases as case_ops
            from hx.memory import harness as rule_ops

            if r is not None and r.status != "retired":
                msg = rule_ops.revise_rule(self.store, arg, retire=True, why="retired by user")
            elif self.store.get_case(arg) is not None:
                msg = case_ops.update_case(self.store, arg, retire=True)
            else:
                return False, f"no live rule or case {arg!r}"
            self.bus.emit("memory_write", actions=[msg], action="forget", touched=[arg])
            return True, msg
        return False, f"unknown command {cmd!r}"


# ---------------------------------------------------------------------------- HTTP
def make_handler(app: App, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            return

        # -- helpers
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts   # blocks DNS rebinding

        def _auth_ok(self, query: dict[str, list[str]]) -> bool:
            tok = self.headers.get("X-HX-Token") or (query.get("token") or [""])[0]
            return secrets.compare_digest(tok, app.token)

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 1_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- routes
        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._send(403, b"bad host", "text/plain")
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                html = (WEB_DIR / "index.html").read_text(encoding="utf-8").replace("__HX_TOKEN__", app.token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if not url.path.startswith("/api/"):
                return self._send(404, b"not found", "text/plain")
            if not self._auth_ok(q):
                return self._send(401, b"unauthorized", "text/plain")
            if url.path == "/api/state":
                return self._json(200, app.state())
            if url.path == "/api/memory":
                return self._json(200, app.memory())
            if url.path == "/api/traces":
                return self._json(200, app.traces())
            if url.path.startswith("/api/trace/"):
                t = app.store.get_trace(url.path.rsplit("/", 1)[-1])
                return self._json(200 if t else 404, json.loads(t.model_dump_json()) if t else {"error": "not found"})
            if url.path == "/api/events":
                return self._stream(q)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return self._send(403, b"bad host", "text/plain")
            url = urlparse(self.path)
            if not self._auth_ok(parse_qs(url.query)):
                return self._send(401, b"unauthorized", "text/plain")
            body = self._body()
            if url.path == "/api/chat":
                ok, msg = app.chat(str(body.get("text") or ""))
            elif url.path == "/api/command":
                ok, msg = app.command(str(body.get("cmd") or ""), str(body.get("arg") or "").strip())
            elif url.path == "/api/confirm":
                ok = app.ui.answer(str(body.get("id") or ""), str(body.get("answer") or "no"))
                msg = "ok" if ok else "no such request"
            else:
                return self._send(404, b"not found", "text/plain")
            return self._json(200 if ok else 409, {"ok": ok, "message": msg})

        def _stream(self, q: dict[str, list[str]]) -> None:
            last = int(self.headers.get("Last-Event-ID") or (q.get("since") or ["0"])[0] or 0)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 1500\n\n")
                self.wfile.flush()
                while True:
                    events = app.bus.since(last, timeout=15)
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                    for e in events:
                        last = e["id"]
                        payload = json.dumps(e, default=str)
                        self.wfile.write(f"id: {e['id']}\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def serve(agent: Agent, store: Store, bus: EventBus, ui: WebUI, settings: Settings,
          port: int = 8765, open_browser: bool = True) -> None:
    app = App(agent, store, bus, ui, settings)
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(app, p))
            port = p
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {port}..{port + 19}")
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"hx ui running at {url}  (memory: {store.name}, cwd: {agent.ctx.workdir})")
    print("Ctrl-C to stop.")
    bus.emit("server_start", url=url)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
