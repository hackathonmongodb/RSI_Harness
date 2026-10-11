"""Tiny mflix NL set so run_baseline works before A's ~42-task file lands."""

from hx.models import Task

FALLBACK_TASKS = [
    Task(
        task_id="mflix_top_commented_2015",
        nl="Show me the genres and runtime of 10 movies from 2015 that have the most comments",
        split="train",
    ),
    Task(
        task_id="mflix_scifi_2010",
        nl="List the titles and release years of 5 sci-fi movies from 2010",
        split="train",
    ),
    Task(
        task_id="mflix_1990s",
        nl="Which movies were released between 1990 and 2000, newest first? Limit 10.",
        split="train",
    ),
    Task(
        task_id="mflix_long_runtimes",
        nl="Find 8 movies with runtime greater than 150 minutes. Return title and runtime only.",
        split="heldout",
    ),
    Task(
        task_id="mflix_tom_hanks",
        nl="Titles of movies where Tom Hanks is in the cast, limit 5",
        split="heldout",
    ),
]
