#!/usr/bin/env python3
"""Create mflix train/held-out tasks + gold. Execute gold; drop errors/empty.

Usage:
  python scripts/make_tasks.py              # write 15 train + 6 held-out
  python scripts/make_tasks.py --review ID  # print task, gold, first 3 rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx.config import get_client
from hx.executor import run_pipeline
from hx.models import Gold, Task

# Hand-crafted first slice for B baselines. Absolute dates only; questions name fields.
# verified=True means human-checked in this commit for the first 10 train + all held-out.
MFLIX_TASKS: list[dict[str, Any]] = [
    # ---- train (15) ----
    {
        "task_id": "mflix_t01_scifi_2010",
        "split": "train",
        "question": (
            "List title and year of 5 movies in genre Sci-Fi with year exactly 2010. "
            "Return fields title, year. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"genres": "Sci-Fi", "year": 2010}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t02_runtime_gt_150",
        "split": "train",
        "question": (
            "Find 8 movies with runtime greater than 150. Return fields title, runtime. Limit 8."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"runtime": {"$gt": 150}}},
            {"$project": {"_id": 0, "title": 1, "runtime": 1}},
            {"$limit": 8},
        ],
    },
    {
        "task_id": "mflix_t03_1990s_newest",
        "split": "train",
        "question": (
            "Movies with year between 1990 and 2000 inclusive, newest year first, limit 10. "
            "Return fields title, year."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["filter", "sort"],
        "verified": True,
        "pipeline": [
            {"$match": {"year": {"$gte": 1990, "$lte": 2000}}},
            {"$sort": {"year": -1, "title": 1}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 10},
        ],
    },
    {
        "task_id": "mflix_t04_tom_hanks",
        "split": "train",
        "question": (
            "Titles of movies where cast contains Tom Hanks, limit 5. Return field title."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["array"],
        "verified": True,
        "pipeline": [
            {"$match": {"cast": "Tom Hanks"}},
            {"$project": {"_id": 0, "title": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t05_genre_counts",
        "split": "train",
        "question": (
            "Count movies per genre after unwinding genres. Return fields genre, count. "
            "Sort by count descending, limit 5."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["group", "array"],
        "verified": True,
        "pipeline": [
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$limit": 5},
            {"$project": {"_id": 0, "genre": "$_id", "count": 1}},
        ],
    },
    {
        "task_id": "mflix_t06_usa_english_2000",
        "split": "train",
        "question": (
            "Movies with countries containing USA, languages containing English, year 2000. "
            "Return fields title, year. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter", "array"],
        "verified": True,
        "pipeline": [
            {"$match": {"countries": "USA", "languages": "English", "year": 2000}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t07_imdb_rating_ge_8",
        "split": "train",
        "question": (
            "Movies with imdb.rating greater than or equal to 8.5 and year 1994. "
            "Return fields title, imdb_rating. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["nested", "filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"imdb.rating": {"$gte": 8.5}, "year": 1994}},
            {"$project": {"_id": 0, "title": 1, "imdb_rating": "$imdb.rating"}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t08_awards_wins",
        "split": "train",
        "question": (
            "Movies with awards.wins greater than 10 and year 2010. "
            "Return fields title, wins. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["nested", "filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"awards.wins": {"$gt": 10}, "year": 2010}},
            {"$project": {"_id": 0, "title": 1, "wins": "$awards.wins"}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t09_released_2015",
        "split": "train",
        "question": (
            "Movies with released on or after 2015-01-01T00:00:00Z and before 2016-01-01T00:00:00Z. "
            "Return fields title, released. Sort by released ascending, limit 5."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["date", "sort"],
        "verified": True,
        "pipeline": [
            {
                "$match": {
                    "released": {
                        "$gte": {"$date": "2015-01-01T00:00:00Z"},
                        "$lt": {"$date": "2016-01-01T00:00:00Z"},
                    }
                }
            },
            {"$sort": {"released": 1, "title": 1}},
            {"$project": {"_id": 0, "title": 1, "released": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t10_lookup_comment_counts_2015",
        "split": "train",
        "question": (
            "For 3 movies from year 2015 with num_mflix_comments >= 1, join comments on movie_id "
            "and return fields title, comment_count where comment_count is the size of the joined "
            "array. Sort by num_mflix_comments descending, limit 3."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["lookup", "sort"],
        "verified": True,
        "pipeline": [
            {"$match": {"year": 2015, "num_mflix_comments": {"$gte": 1}}},
            {"$sort": {"num_mflix_comments": -1, "title": 1}},
            {"$limit": 3},
            {
                "$lookup": {
                    "from": "comments",
                    "localField": "_id",
                    "foreignField": "movie_id",
                    "as": "cmts",
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "comment_count": {"$size": "$cmts"},
                }
            },
        ],
    },
    {
        "task_id": "mflix_t11_comedy_runtime",
        "split": "train",
        "question": (
            "Comedy movies (genres contains Comedy) with runtime between 90 and 100 inclusive, "
            "year 2005. Return fields title, runtime. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter", "array"],
        "verified": False,
        "pipeline": [
            {
                "$match": {
                    "genres": "Comedy",
                    "runtime": {"$gte": 90, "$lte": 100},
                    "year": 2005,
                }
            },
            {"$project": {"_id": 0, "title": 1, "runtime": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t12_avg_runtime_by_year",
        "split": "train",
        "question": (
            "Average runtime of movies for years 2008, 2009, and 2010. Return fields year, "
            "avg_runtime sorted by year ascending."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["group"],
        "verified": False,
        "pipeline": [
            {"$match": {"year": {"$in": [2008, 2009, 2010]}, "runtime": {"$type": "int"}}},
            {"$group": {"_id": "$year", "avg_runtime": {"$avg": "$runtime"}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "avg_runtime": 1}},
        ],
    },
    {
        "task_id": "mflix_t13_directors_spielberg",
        "split": "train",
        "question": (
            "Movies where directors contains Steven Spielberg and year >= 1990. "
            "Return fields title, year. Sort by year descending, limit 5."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["array", "sort"],
        "verified": False,
        "pipeline": [
            {"$match": {"directors": "Steven Spielberg", "year": {"$gte": 1990}}},
            {"$sort": {"year": -1, "title": 1}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t14_theater_ca",
        "split": "train",
        "question": (
            "Theaters in location.address.state CA. Return fields theaterId, city. Limit 5."
        ),
        "collection": "theaters",
        "ordered": False,
        "tags": ["nested", "filter"],
        "verified": False,
        "pipeline": [
            {"$match": {"location.address.state": "CA"}},
            {
                "$project": {
                    "_id": 0,
                    "theaterId": 1,
                    "city": "$location.address.city",
                }
            },
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_t15_top_commented_2014",
        "split": "train",
        "question": (
            "For year 2014, return the 5 movies with highest num_mflix_comments. "
            "Return fields title, num_mflix_comments. Sort by num_mflix_comments descending."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["filter", "sort"],
        "verified": False,
        "pipeline": [
            {"$match": {"year": 2014}},
            {"$sort": {"num_mflix_comments": -1, "title": 1}},
            {"$project": {"_id": 0, "title": 1, "num_mflix_comments": 1}},
            {"$limit": 5},
        ],
    },
    # ---- held-out (6) ----
    {
        "task_id": "mflix_h01_action_2012",
        "split": "heldout",
        "question": (
            "Action movies (genres contains Action) with year 2012. "
            "Return fields title, year. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"genres": "Action", "year": 2012}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_h02_countries_uk_2000",
        "split": "heldout",
        "question": (
            "Movies with countries containing UK and year exactly 2000. "
            "Return fields title, year. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["array", "filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"countries": "UK", "year": 2000}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_h03_languages_french",
        "split": "heldout",
        "question": (
            "Movies with languages containing French and year 1995. "
            "Return fields title, year. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["array", "filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"languages": "French", "year": 1995}},
            {"$project": {"_id": 0, "title": 1, "year": 1}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_h04_imdb_votes",
        "split": "heldout",
        "question": (
            "Movies with imdb.votes greater than 500000 and year between 1990 and 1999. "
            "Return fields title, votes. Sort by votes descending, limit 5."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["nested", "sort"],
        "verified": True,
        "pipeline": [
            {
                "$match": {
                    "imdb.votes": {"$gt": 500000},
                    "year": {"$gte": 1990, "$lte": 1999},
                }
            },
            {"$sort": {"imdb.votes": -1, "title": 1}},
            {"$project": {"_id": 0, "title": 1, "votes": "$imdb.votes"}},
            {"$limit": 5},
        ],
    },
    {
        "task_id": "mflix_h05_lookup_one_comment",
        "split": "heldout",
        "question": (
            "Take 2 movies from year 2000 with num_mflix_comments >= 1 sorted by "
            "num_mflix_comments descending, $lookup comments, and return fields title, "
            "first_commenter (name of the first joined comment) and comment_count."
        ),
        "collection": "movies",
        "ordered": True,
        "tags": ["lookup"],
        "verified": True,
        "pipeline": [
            {"$match": {"year": 2000, "num_mflix_comments": {"$gte": 1}}},
            {"$sort": {"num_mflix_comments": -1, "title": 1}},
            {"$limit": 2},
            {
                "$lookup": {
                    "from": "comments",
                    "localField": "_id",
                    "foreignField": "movie_id",
                    "as": "cmts",
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "first_commenter": {"$arrayElemAt": ["$cmts.name", 0]},
                    "comment_count": {"$size": "$cmts"},
                }
            },
        ],
    },
    {
        "task_id": "mflix_h06_rated_pg13_2011",
        "split": "heldout",
        "question": (
            "Movies with rated equal to PG-13 and year 2011. "
            "Return fields title, rated. Limit 5."
        ),
        "collection": "movies",
        "ordered": False,
        "tags": ["filter"],
        "verified": True,
        "pipeline": [
            {"$match": {"rated": "PG-13", "year": 2011}},
            {"$project": {"_id": 0, "title": 1, "rated": 1}},
            {"$limit": 5},
        ],
    },
]


def _decode_pipeline(pipeline: list[dict]) -> list[dict]:
    """Expand Extended JSON date literals for pymongo."""
    from bson.json_util import loads, dumps

    return loads(dumps(pipeline))


def write_tasks(dataset: str = "sample_mflix") -> None:
    admin = get_client("admin")
    tasks_coll = admin["hx_eval"]["tasks"]
    gold_coll = admin["hx_eval"]["gold"]

    kept = 0
    dropped = 0
    for raw in MFLIX_TASKS:
        pipe = _decode_pipeline(raw["pipeline"])
        try:
            rows = run_pipeline(dataset, raw["collection"], pipe, max_ms=15000, row_cap=500)
        except Exception as err:  # noqa: BLE001
            print(f"DROP {raw['task_id']}: exec error: {err}")
            dropped += 1
            continue
        if not rows:
            print(f"DROP {raw['task_id']}: empty result")
            dropped += 1
            continue

        task = Task(
            task_id=raw["task_id"],
            dataset=dataset,
            split=raw["split"],
            question=raw["question"],
            collection=raw["collection"],
            ordered=bool(raw.get("ordered")),
            tags=list(raw.get("tags") or []),
        )
        gold = Gold(
            task_id=raw["task_id"],
            pipeline=pipe,
            verified=bool(raw.get("verified")),
        )
        tasks_coll.update_one(
            {"task_id": task.task_id},
            {"$set": task.model_dump()},
            upsert=True,
        )
        gold_coll.update_one(
            {"task_id": gold.task_id},
            {"$set": gold.model_dump()},
            upsert=True,
        )
        print(f"OK {task.task_id} split={task.split} n={len(rows)} verified={gold.verified}")
        kept += 1

    n_train = tasks_coll.count_documents({"dataset": dataset, "split": "train"})
    n_hold = tasks_coll.count_documents({"dataset": dataset, "split": "heldout"})
    n_ver = gold_coll.count_documents(
        {"task_id": {"$regex": "^mflix_"}, "verified": True}
    )
    print(f"done kept={kept} dropped={dropped} train={n_train} heldout={n_hold} verified_gold={n_ver}")


def review(task_id: str) -> None:
    admin = get_client("admin")
    task = admin["hx_eval"]["tasks"].find_one({"task_id": task_id})
    gold = admin["hx_eval"]["gold"].find_one({"task_id": task_id})
    if not task or not gold:
        raise SystemExit(f"missing task/gold for {task_id}")
    print("TASK:", task["question"])
    print("COLLECTION:", task["collection"], "ORDERED:", task.get("ordered"))
    print("PIPELINE:", gold["pipeline"])
    rows = run_pipeline(
        task["dataset"], task["collection"], gold["pipeline"], max_ms=15000, row_cap=3
    )
    print("FIRST ROWS:")
    for r in rows[:3]:
        print(" ", r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--review", metavar="TASK_ID")
    args = p.parse_args()
    if args.review:
        review(args.review)
    else:
        write_tasks()


if __name__ == "__main__":
    main()
