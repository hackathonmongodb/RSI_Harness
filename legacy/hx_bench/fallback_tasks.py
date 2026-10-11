from hx.models import Task

FALLBACK_TASKS = [
    Task(
        task_id="mflix_top_commented_2015",
        dataset="sample_mflix",
        split="train",
        question="Show the genres and runtime of 10 movies from 2015 that have the most comments. Return genres, runtime.",
        collection="movies",
        ordered=True,
        tags=["filter", "sort"],
    ),
    Task(
        task_id="mflix_scifi_2010",
        dataset="sample_mflix",
        split="train",
        question="List title and year of 5 sci-fi movies from 2010. Return title, year.",
        collection="movies",
        tags=["filter"],
    ),
    Task(
        task_id="mflix_1990s",
        dataset="sample_mflix",
        split="train",
        question="Movies released between 1990 and 2000, newest first, limit 10. Return title, year.",
        collection="movies",
        ordered=True,
        tags=["date", "sort"],
    ),
    Task(
        task_id="mflix_long_runtimes",
        dataset="sample_mflix",
        split="heldout",
        question="Find 8 movies with runtime greater than 150. Return title and runtime only.",
        collection="movies",
        tags=["filter"],
    ),
    Task(
        task_id="mflix_tom_hanks",
        dataset="sample_mflix",
        split="heldout",
        question="Titles of movies where Tom Hanks is in the cast, limit 5. Return title.",
        collection="movies",
        tags=["array"],
    ),
]
