"""Accountant ask: MongoDB AI (Voyage) picks a metric query → Atlas → plain English.

Uses ai.mongodb.com embeddings/rerank for retrieval. When harness accuracy is
below target, the agent may trigger Llama-backed RSI evolve (no UI button).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from bson.json_util import dumps as bson_dumps
import json5

from rsi_harness.db import db
from rsi_harness.mflix_db import mflix
from rsi_harness.mflix_rsi import maybe_auto_evolve
from rsi_harness.mongo_ai import MongoAIError, rerank

# Catalog of safe, read-only business questions the accountant can ask.
MetricFn = Callable[[], list[dict[str, Any]]]


def _avg_score_by_version() -> list[dict[str, Any]]:
    return list(
        db()["eval_runs"].aggregate(
            [
                {
                    "$group": {
                        "_id": "$harness_version",
                        "avg_score": {"$avg": "$score"},
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"avg_score": 1}},
                {
                    "$project": {
                        "_id": 0,
                        "harness_version": "$_id",
                        "avg_score": 1,
                        "n": 1,
                    }
                },
            ]
        )
    )


def _lowest_eval_runs() -> list[dict[str, Any]]:
    return list(
        db()["eval_runs"]
        .find({}, {"_id": 0, "harness_version": 1, "task": 1, "score": 1, "notes": 1})
        .sort("score", 1)
        .limit(8)
    )


def _production_harness() -> list[dict[str, Any]]:
    return list(
        db()["harness_versions"]
        .find(
            {"status": "production"},
            {
                "_id": 0,
                "version": 1,
                "status": 1,
                "avg_score": 1,
                "mutation_reason": 1,
                "rules": 1,
            },
        )
        .sort("created_at", -1)
        .limit(5)
    )


def _guardrail_by_version() -> list[dict[str, Any]]:
    return list(
        db()["eval_runs"].aggregate(
            [
                {
                    "$group": {
                        "_id": "$harness_version",
                        "avg_score": {"$avg": "$score"},
                        "guardrail_violations": {
                            "$sum": {"$ifNull": ["$guardrail_violations", 0]}
                        },
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
                {
                    "$project": {
                        "_id": 0,
                        "harness_version": "$_id",
                        "avg_score": 1,
                        "guardrail_violations": 1,
                        "n": 1,
                    }
                },
            ]
        )
    )


def _memories_long_horizon() -> list[dict[str, Any]]:
    return list(
        db()["memories"]
        .find(
            {"tags": {"$in": ["long-horizon", "demo"]}},
            {"_id": 0, "tier": 1, "text": 1, "tags": 1},
        )
        .sort("created_at", -1)
        .limit(20)
    )


def _harness_count() -> list[dict[str, Any]]:
    n = db()["harness_versions"].count_documents({})
    prod = db()["harness_versions"].count_documents({"status": "production"})
    return [{"total_harness_versions": n, "production_count": prod}]


def _movies_by_genre() -> list[dict[str, Any]]:
    return list(
        mflix()["movies"].aggregate(
            [
                {"$unwind": "$genres"},
                {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 8},
                {"$project": {"_id": 0, "genre": "$_id", "count": 1}},
            ]
        )
    )


def _movies_tom_hanks() -> list[dict[str, Any]]:
    return list(
        mflix()["movies"]
        .find({"cast": "Tom Hanks"}, {"_id": 0, "title": 1, "year": 1})
        .limit(8)
    )


def _movies_high_imdb_1994() -> list[dict[str, Any]]:
    return list(
        mflix()["movies"].aggregate(
            [
                {"$match": {"imdb.rating": {"$gte": 8.5}, "year": 1994}},
                {"$project": {"_id": 0, "title": 1, "imdb_rating": "$imdb.rating"}},
                {"$limit": 8},
            ]
        )
    )


def _movies_scifi_2010() -> list[dict[str, Any]]:
    return list(
        mflix()["movies"]
        .find({"genres": "Sci-Fi", "year": 2010}, {"_id": 0, "title": 1, "year": 1})
        .limit(8)
    )


def _movies_count() -> list[dict[str, Any]]:
    return [{"movies": mflix()["movies"].estimated_document_count(),
             "comments": mflix()["comments"].estimated_document_count()}]


METRICS: list[dict[str, Any]] = [
    {
        "id": "avg_score_by_version",
        "description": (
            "what is the average eval score by harness version lowest first "
            "mean performance metrics ranking versions by score only"
        ),
        "mongosh": (
            "db.eval_runs.aggregate([{$group:{_id:'$harness_version',avg_score:{$avg:'$score'},n:{$sum:1}}},{$sort:{avg_score:1}}])"
        ),
        "collection": "eval_runs",
        "run": _avg_score_by_version,
        "format": "avg_score",
    },
    {
        "id": "lowest_evals",
        "description": (
            "lowest scoring eval runs tasks notes worst failures bottom scores"
        ),
        "mongosh": "db.eval_runs.find().sort({score:1}).limit(8)",
        "collection": "eval_runs",
        "run": _lowest_eval_runs,
        "format": "lowest",
    },
    {
        "id": "production_harness",
        "description": (
            "current production harness rules mutation reason active version"
        ),
        "mongosh": "db.harness_versions.find({status:'production'}).sort({created_at:-1})",
        "collection": "harness_versions",
        "run": _production_harness,
        "format": "production",
    },
    {
        "id": "guardrails",
        "description": (
            "guardrail violations and safety flags grouped by harness version "
            "with average scores for compliance reporting"
        ),
        "mongosh": (
            "db.eval_runs.aggregate([{$group:{_id:'$harness_version',avg_score:{$avg:'$score'},guardrail_violations:{$sum:'$guardrail_violations'}}}])"
        ),
        "collection": "eval_runs",
        "run": _guardrail_by_version,
        "format": "guardrails",
    },
    {
        "id": "memories",
        "description": "memories tagged long-horizon or demo notes reminders",
        "mongosh": "db.memories.find({tags:{$in:['long-horizon','demo']}})",
        "collection": "memories",
        "run": _memories_long_horizon,
        "format": "memories",
    },
    {
        "id": "harness_count",
        "description": "how many harness versions exist production count inventory",
        "mongosh": "db.harness_versions.countDocuments({})",
        "collection": "harness_versions",
        "run": _harness_count,
        "format": "count",
    },
    # ---- sample_mflix (movie database) ----
    {
        "id": "mflix_genre_counts",
        "description": (
            "movie genres count how many movies per genre drama comedy action "
            "sample_mflix films catalog breakdown"
        ),
        "mongosh": "db.movies.aggregate([{$unwind:'$genres'},{$group:{_id:'$genres',count:{$sum:1}}},{$sort:{count:-1}}])",
        "collection": "movies",
        "run": _movies_by_genre,
        "format": "mflix_genres",
    },
    {
        "id": "mflix_tom_hanks",
        "description": (
            "Tom Hanks movies cast films starring Tom Hanks titles years sample_mflix"
        ),
        "mongosh": "db.movies.find({cast:'Tom Hanks'},{title:1,year:1}).limit(8)",
        "collection": "movies",
        "run": _movies_tom_hanks,
        "format": "mflix_titles",
    },
    {
        "id": "mflix_imdb_1994",
        "description": (
            "highest imdb rating movies from 1994 rating above 8.5 best films that year"
        ),
        "mongosh": "db.movies.find({year:1994,'imdb.rating':{$gte:8.5}},{title:1,'imdb.rating':1})",
        "collection": "movies",
        "run": _movies_high_imdb_1994,
        "format": "mflix_imdb",
    },
    {
        "id": "mflix_scifi_2010",
        "description": (
            "sci-fi science fiction movies from year 2010 titles sample_mflix"
        ),
        "mongosh": "db.movies.find({genres:'Sci-Fi',year:2010},{title:1,year:1}).limit(8)",
        "collection": "movies",
        "run": _movies_scifi_2010,
        "format": "mflix_titles",
    },
    {
        "id": "mflix_counts",
        "description": (
            "how many movies and comments are in the movie database sample_mflix size"
        ),
        "mongosh": "db.movies.estimatedDocumentCount()",
        "collection": "movies",
        "run": _movies_count,
        "format": "mflix_counts",
    },
]


def _serialize(rows: list[dict[str, Any]]) -> list[Any]:
    return json5.loads(bson_dumps(rows))


def _fmt_num(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _interpret(fmt: str, rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not rows:
        return (
            "No matching data was found for that question. Try reseeding the sandbox first.",
            [],
        )

    if fmt == "avg_score":
        highlights = [
            f"Version {r.get('harness_version')}: average score {_fmt_num(r.get('avg_score'))} "
            f"(from {r.get('n')} runs)"
            for r in rows[:5]
        ]
        best = max(rows, key=lambda r: float(r.get("avg_score") or 0))
        answer = (
            f"Across harness versions, the strongest average score is "
            f"{_fmt_num(best.get('avg_score'))} on version {best.get('harness_version')}."
        )
        return answer, highlights

    if fmt == "lowest":
        highlights = [
            f"{r.get('task') or 'task'}: score {_fmt_num(r.get('score'))} "
            f"(harness v{r.get('harness_version')})"
            for r in rows[:5]
        ]
        worst = rows[0]
        answer = (
            f"The weakest eval so far is “{worst.get('task') or 'a task'}” "
            f"with score {_fmt_num(worst.get('score'))}."
        )
        return answer, highlights

    if fmt == "production":
        cur = rows[0]
        rules = cur.get("rules") or []
        highlights = [str(x) for x in rules[:5]]
        answer = (
            f"Production harness is v{cur.get('version')} "
            f"(avg score {_fmt_num(cur.get('avg_score'))}). "
            f"{cur.get('mutation_reason') or ''}".strip()
        )
        return answer, highlights

    if fmt == "guardrails":
        highlights = [
            f"Version {r.get('harness_version')}: avg {_fmt_num(r.get('avg_score'))}, "
            f"guardrail flags {r.get('guardrail_violations', 0)}"
            for r in rows[:5]
        ]
        answer = "Here are average scores and guardrail flags by harness version."
        return answer, highlights

    if fmt == "memories":
        highlights = [str(r.get("text") or "")[:160] for r in rows[:5] if r.get("text")]
        answer = f"Found {len(rows)} memory note(s) tagged long-horizon or demo."
        return answer, highlights

    if fmt == "count":
        r = rows[0]
        answer = (
            f"There are {r.get('total_harness_versions')} harness version(s) total, "
            f"including {r.get('production_count')} in production."
        )
        return answer, []

    if fmt == "mflix_genres":
        highlights = [f"{r.get('genre')}: {r.get('count')} movies" for r in rows[:8]]
        top = rows[0]
        answer = (
            f"In the movie database, the largest genre is {top.get('genre')} "
            f"with {top.get('count')} films."
        )
        return answer, highlights

    if fmt == "mflix_titles":
        highlights = [
            f"{r.get('title')} ({r.get('year')})" if r.get("year") is not None else str(r.get("title"))
            for r in rows[:8]
        ]
        answer = f"Here are {len(rows)} matching movie title(s) from sample_mflix."
        return answer, highlights

    if fmt == "mflix_imdb":
        highlights = [
            f"{r.get('title')}: IMDb {_fmt_num(r.get('imdb_rating'))}" for r in rows[:8]
        ]
        answer = f"Found {len(rows)} highly rated movie(s) from 1994 (IMDb ≥ 8.5)."
        return answer, highlights

    if fmt == "mflix_counts":
        r = rows[0]
        answer = (
            f"The movie database has about {r.get('movies')} movies and "
            f"{r.get('comments')} comments."
        )
        return answer, []

    return f"Returned {len(rows)} row(s).", [str(rows[0])[:160]]


def _pick_metric(question: str) -> dict[str, Any]:
    docs = [m["description"] for m in METRICS]
    ranked = rerank(question, docs, top_k=1)
    if not ranked:
        raise MongoAIError("MongoDB AI rerank returned no match")
    idx = int(ranked[0]["index"])
    score = float(ranked[0].get("relevance_score") or 0)
    metric = METRICS[idx]
    metric = {**metric, "relevance_score": score}
    return metric


def ask(question: str) -> dict[str, Any]:
    """Match question via MongoDB AI rerank, run Atlas query, format for an accountant."""
    q = (question or "").strip()
    if not q:
        raise ValueError("question required")

    t0 = time.perf_counter()
    metric = _pick_metric(q)
    rows_raw = metric["run"]()
    rows = _serialize(rows_raw)
    answer, highlights = _interpret(metric["format"], rows_raw if rows_raw else [])

    # Agent owns evolution — improve until target / plateau / max gens when needed
    evolution = maybe_auto_evolve()
    ms = int((time.perf_counter() - t0) * 1000)

    return {
        "question": q,
        "answer": answer,
        "highlights": highlights,
        "metric_id": metric["id"],
        "relevance_score": metric.get("relevance_score"),
        "mongosh": metric["mongosh"],
        "collection": metric["collection"],
        "operation": "aggregate_or_find",
        "rows": rows[:40],
        "row_count": len(rows),
        "latency_ms": ms,
        "provider": "ai.mongodb.com",
        "evolution": evolution,
    }
