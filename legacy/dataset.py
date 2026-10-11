"""Load eval tasks from data/tasks.jsonl (Stream A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASKS_PATH = ROOT / "data" / "tasks.jsonl"

SAMPLE_TASKS: list[dict[str, Any]] = [
    {
        "id": "mflix_count",
        "prompt": "How many movies are in the sample_mflix.movies collection?",
        "reference": "A correct answer reports the document count of movies (approx tens of thousands).",
    },
    {
        "id": "mflix_genres",
        "prompt": "Which genre appears most often among movies?",
        "reference": "Should name the top genre by frequency (often Drama) with optional count.",
    },
    {
        "id": "mflix_hanks",
        "prompt": "List a few movies starring Tom Hanks.",
        "reference": "Should return real Tom Hanks titles from the dataset (e.g. Forrest Gump, Toy Story).",
    },
    {
        "id": "mflix_scifi_2010",
        "prompt": "Find science fiction movies from 2010.",
        "reference": "Titles should be 2010 films tagged Sci-Fi / science fiction.",
    },
    {
        "id": "mflix_imdb_1994",
        "prompt": "What are the highest IMDb-rated movies from 1994?",
        "reference": "Should prefer 1994 films with high imdb.rating (e.g. Shawshank / Pulp Fiction era).",
    },
]


def ensure_sample_tasks(path: Path = TASKS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            for row in SAMPLE_TASKS:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_tasks(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else ensure_sample_tasks()
    if not p.exists():
        ensure_sample_tasks(p)
    tasks: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"{p}:{line_no} needs id and prompt")
            tasks.append(row)
    return tasks
