from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "valid_early_stop_delta_diffusion.py"
RUN_PATH = EXPERIMENT_DIR / "run_valid_early_stop_delta_diffusion_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_valid_early_stop_delta_diffusion_pilot.sh"


def test_early_stop_supervision_is_train_only_and_generation_unchanged() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def local_changed_orders" in source
    assert "def prefix_actions" in source
    assert "def valid_prefix_candidates" in source
    assert "def select_early_stop_pairs" in source
    assert '"train_only_valid_early_stop_supervision": True' in source
    assert '"train_only_property_oracle_for_trajectory_labels": True' in source
    assert '"train_only_rdkit_validity_for_trajectory_labels": True' in source
    assert '"generation_target_access": False' in source
    assert '"generation_rdkit_validity_access": False' in source
    assert '"posthoc_molecule_repair": False' in source
    assert "candidate_rows, metrics = delta.evaluate(" in source
    assert "model, representation, vocabulary, validation_pairs, args, device" in source


def test_trajectory_evidence_gate_precedes_gpu_training() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    gate_index = source.index("if trajectory_failures:")
    train_index = source.index("history = delta.train_model(")
    assert gate_index < train_index
    assert '"early_stop_coverage"' in source
    assert '"selected_strict_rate"' in source
    assert '"evaluation": None' in source


def test_runner_is_matched_exact_n20_and_bounded() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_EARLY_STOP_DIFFUSION_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_EARLY_STOP_DIFFUSION_VALIDATION_LIMIT:-20}"' in run_source
    assert "--gate-early-stop-coverage 0.20" in run_source
    assert "--gate-selected-strict-rate 0.80" in run_source
    assert "--num-attempts 20" in run_source
    assert "--gate-validity 0.95" in run_source
    assert "--gate-mean-unique-valid 10" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=8G" in submit_source
    assert "00:25:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
