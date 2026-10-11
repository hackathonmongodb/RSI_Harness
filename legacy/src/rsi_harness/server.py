from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rsi_harness.ask import ask as accountant_ask
from rsi_harness.db import ping
from rsi_harness.evolve import evolve_once
from rsi_harness.harness import active, list_versions
from rsi_harness.mflix_rsi import run_mflix_rsi
from rsi_harness.nl_query import run as nl_run
from rsi_harness.seed import seed

WEB = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="RSI Harness", version="0.1.0")


class NLBody(BaseModel):
    query: str


class AskBody(BaseModel):
    question: str


class EvolveBody(BaseModel):
    focus: str | None = None


@app.get("/api/health")
def health():
    try:
        ping()
        return {"ok": True, "mongo": "up"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/seed")
def api_seed():
    try:
        return seed()
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/api/harness")
def api_harness():
    try:
        cur = active()
        cur["_id"] = str(cur.get("_id"))
        versions = []
        for v in list_versions():
            v["_id"] = str(v.get("_id"))
            versions.append(v)
        return {"active": cur, "versions": versions}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.post("/api/nl-query")
def api_nl(body: NLBody):
    if not body.query.strip():
        raise HTTPException(400, "query required")
    try:
        return nl_run(body.query.strip())
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.post("/api/ask")
def api_ask(body: AskBody):
    """Accountant mode: query Atlas, then explain results in plain English."""
    if not body.question.strip():
        raise HTTPException(400, "question required")
    try:
        return accountant_ask(body.question.strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.post("/api/evolve")
def api_evolve(body: EvolveBody):
    try:
        return evolve_once(body.focus)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


class MflixRsiBody(BaseModel):
    generations: int | None = None


@app.post("/api/mflix-rsi")
def api_mflix_rsi(body: MflixRsiBody):
    """Internal agent evolve hook (not exposed in UI). Force-run with stop conditions."""
    gens = body.generations
    if gens is not None:
        gens = max(1, min(int(gens), 12))
    try:
        return run_mflix_rsi(gens, force=True)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
