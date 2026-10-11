"""Web UI server: auth, host check, chat -> events, approvals through the browser."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from hx.agent import Agent
from hx.llm import Reply
from hx.server import App, EventBus, WebUI, make_handler
from tests.conftest import FakeLLM, call


@pytest.fixture
def served(settings, store):
    bus = EventBus()
    ui = WebUI(bus)
    llm = FakeLLM([Reply(tool_calls=[call("bash", command="echo hi")]), Reply(content="done!")])
    agent = Agent(settings=settings.with_(auto_approve=False), llm=llm, store=store, ui=ui)
    app = App(agent, store, bus, ui, settings)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, port
    srv.shutdown()


def req(port, path, token=None, body=None, host=None):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                               data=json.dumps(body).encode() if body is not None else None)
    if token:
        r.add_header("X-HX-Token", token)
    r.add_header("Content-Type", "application/json")
    if host:
        r.add_header("Host", host)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_index_embeds_token_and_api_requires_it(served):
    app, port = served
    code, html = req(port, "/")
    assert code == 200 and app.token.encode() in html
    assert req(port, "/api/state")[0] == 401
    code, body = req(port, "/api/state", app.token)
    assert code == 200 and json.loads(body)["busy"] is False


def test_bad_host_rejected(served):
    app, port = served
    assert req(port, "/api/state", app.token, host="evil.example:80")[0] == 403


def test_chat_runs_turn_and_waits_for_browser_approval(served):
    app, port = served
    code, body = req(port, "/api/chat", app.token, {"text": "say hi"})
    assert code == 200 and json.loads(body)["ok"]
    # the bash call blocks on an approval request
    deadline = time.time() + 5
    pending = None
    while time.time() < deadline and pending is None:
        pending = next((e for e in app.bus.since(0, 0.1) if e["kind"] == "confirm_request"), None)
    assert pending is not None
    assert req(port, "/api/chat", app.token, {"text": "again"})[0] == 409     # busy
    code, body = req(port, "/api/confirm", app.token, {"id": pending["data"]["id"], "answer": "yes"})
    assert code == 200, body
    deadline = time.time() + 5
    while app.busy and time.time() < deadline:
        time.sleep(0.05)
    kinds = [e["kind"] for e in app.bus.since(0, 0.1)]
    for k in ("user_message", "turn_start", "assemble", "llm_start", "llm_end", "tool_start", "tool_end", "final", "turn_end"):
        assert k in kinds, k
    tool_end = next(e for e in app.bus.since(0, 0.1) if e["kind"] == "tool_end")
    assert tool_end["data"]["ok"] and "hi" in tool_end["data"]["output"]


def test_memory_endpoint_renders_markdown(served):
    app, port = served
    from hx.memory import cases as case_ops

    case_ops.file_case(app.store, "T", "a description", "**Symptom:** x", case_id="c1")
    data = json.loads(req(port, "/api/memory", app.token)[1])
    assert data["harness_md"].startswith("# Harness v0")
    c = data["cases"][0]
    assert c["file"] == "cases/c1.md" and c["markdown"].startswith("---\ncase_id: \"c1\"")
