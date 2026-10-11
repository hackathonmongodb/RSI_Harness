from __future__ import annotations

from hx.overview import build_overview
from tests.mocks import STORE, reset_store


def setup_function() -> None:
    reset_store()


def test_empty_harness_zeros() -> None:
    out = build_overview("no-such")
    assert out["n_reports"] == 0
    assert out["missing_component_count"] == 0
    assert out["by_component"] == []
    assert "$facet" in out["_facet_pipeline"][1]


def test_counts_match_injected_reports() -> None:
    STORE.reports.extend(
        [
            {
                "harness_id": "h0",
                "implicated_component_id": "rule.types",
                "implicated_kind": "rule",
                "error_type": "wrong_result",
                "missing_component": False,
            },
            {
                "harness_id": "h0",
                "implicated_component_id": "rule.types",
                "implicated_kind": "rule",
                "error_type": "parse",
                "missing_component": False,
            },
            {
                "harness_id": "h0",
                "implicated_component_id": None,
                "implicated_kind": None,
                "error_type": "wrong_result",
                "missing_component": True,
            },
            {
                "harness_id": "other",
                "implicated_component_id": "rule.types",
                "implicated_kind": "rule",
                "error_type": "timeout",
                "missing_component": False,
            },
        ]
    )
    out = build_overview("h0")
    assert out["n_reports"] == 3
    assert out["missing_component_count"] == 1
    assert out["by_component"][0]["id"] == "rule.types"
    assert out["by_component"][0]["n"] == 2
