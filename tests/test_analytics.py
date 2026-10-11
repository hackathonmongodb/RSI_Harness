from __future__ import annotations

from hx.analytics import prediction_scores, set_manifests
from hx.models import Edit, Manifest


def _m(
    edit_id: str,
    generation: int,
    pred_fix: list[str],
    actual_fix: list[str],
    pred_reg: list[str] | None = None,
    actual_reg: list[str] | None = None,
) -> Manifest:
    return Manifest(
        edit_id=edit_id,
        generation=generation,
        phase="growth",
        parent_harness_id="h0",
        edit=Edit(op="add", kind="rule", target_component_id="rule.x"),
        root_cause="rc",
        targeted_fix="tf",
        predicted_fixes=pred_fix,
        predicted_regressions=pred_reg or [],
        actual_fixed=actual_fix,
        actual_regressed=actual_reg or [],
    )


def test_precision_recall_known_sets() -> None:
    set_manifests(
        [
            _m("e1", 1, ["a", "b"], ["a"], pred_reg=["x"], actual_reg=["x", "y"]),
        ]
    )
    out = prediction_scores(1)
    assert out["n_manifests"] == 1
    assert out["per_edit"][0]["fix_precision"] == 0.5
    assert out["per_edit"][0]["fix_recall"] == 1.0
    assert out["per_edit"][0]["reg_precision"] == 1.0
    assert out["per_edit"][0]["reg_recall"] == 0.5


def test_empty_predicted_precision_is_none() -> None:
    set_manifests([_m("e2", 2, [], ["a"])])
    out = prediction_scores(2)
    assert out["fix_precision"] is None
    assert out["fix_recall"] == 0.0


def test_empty_actual_recall_is_none() -> None:
    set_manifests([_m("e3", 3, ["a"], [])])
    out = prediction_scores(3)
    assert out["fix_precision"] == 0.0
    assert out["fix_recall"] is None


def test_filters_by_generation() -> None:
    set_manifests([_m("e1", 1, ["a"], ["a"]), _m("e2", 2, ["b"], ["b"])])
    assert prediction_scores(1)["n_manifests"] == 1
    assert prediction_scores(9)["n_manifests"] == 0
