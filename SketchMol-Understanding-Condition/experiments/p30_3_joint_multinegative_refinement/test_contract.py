from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_joint_refinement_contract():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    train = (HERE / "run_train.sh").read_text()
    submit = (HERE / "submit_joint_refinement.sh").read_text()
    collect = (HERE / "collect_joint_gate.py").read_text()
    assert prereg["refinement"]["total_pairs"] == 1200
    assert prereg["refinement"]["negative_type"] == "invalid_corruption only"
    assert "checkpoint-030/adapter" in train
    assert "--epochs 0.5 --learning-rate 2e-6" in train
    assert "editing_baseline_job" in submit
    assert "editing_refined_job" in submit
    assert "denovo_gate_job" in submit
    assert "edit_strict_non_regression" in collect
    assert "RUN_FULL_BUDGET_CURVE" in collect
