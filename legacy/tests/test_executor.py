"""Unit tests for hx.executor.validate_pipeline / run_pipeline."""

from __future__ import annotations

import pytest

from hx.executor import BLOCKED_STAGES, validate_pipeline


def test_validate_ok():
    validate_pipeline([{"$match": {"a": 1}}, {"$limit": 1}])


def test_validate_max_stages():
    pipe = [{"$match": {"a": i}} for i in range(13)]
    with pytest.raises(ValueError, match="max 12"):
        validate_pipeline(pipe)


@pytest.mark.parametrize("stage", sorted(BLOCKED_STAGES))
def test_blocked_stages(stage: str):
    with pytest.raises(ValueError, match="blocked"):
        validate_pipeline([{stage: "movies"}])


def test_not_list():
    with pytest.raises(ValueError, match="list"):
        validate_pipeline({"$match": {}})  # type: ignore[arg-type]
