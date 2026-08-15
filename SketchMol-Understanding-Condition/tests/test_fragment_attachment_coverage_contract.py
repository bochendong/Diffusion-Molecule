from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "audit_fragment_attachment_trajectory_coverage.py"
RUN_PATH = EXPERIMENT_DIR / "run_fragment_attachment_coverage_gate.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_fragment_attachment_coverage_gate.sh"


def test_coverage_audit_uses_train_pairs_only_for_fragment_support() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def pair_fragment_support" in source
    assert "fragments.fragment_splits" in source
    assert "fragments.join_fragments" in source
    assert '"development_target_fragment_access": False' in source
    assert '"train_only_fragment_vocabulary": True' in source
    assert '"requested_accelerator_hours": 0' in source
    assert "for pair in train_pairs:" in source


def test_gate_checks_growth_and_three_property_support_before_training() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert '"overall_coverage"' in source
    assert '"three_property_coverage"' in source
    assert '"growth_task_coverage"' in source
    assert '"exact_reconstruction"' in source
    assert '"unique_target_fragments"' in source
    assert '"train_latent_fragment_attachment_kernel"' in source


def test_runner_is_bounded_cpu_only() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_FRAGMENT_COVERAGE_TRAIN_LIMIT:-1500}"' in run_source
    assert "--gate-growth-task-coverage 0.30" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=4G" in submit_source
    assert "00:12:00" in submit_source
    assert "--gres" not in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
