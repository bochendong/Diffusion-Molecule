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
TRAIN_PATH = EXPERIMENT_DIR / "train_molecular_latent_autoencoder.py"
RUN_PATH = EXPERIMENT_DIR / "run_molecular_latent_autoencoder_v2.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_molecular_latent_autoencoder_v2.sh"


def test_representation_stage_has_no_condition_or_selector_path() -> None:
    source = TRAIN_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"representation_stage_only": True' in source
    assert '"condition_access": False' in source
    assert '"property_oracle_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"benchmark_generation_target_access": False' in source
    assert '"representation_validation_inputs_include_source_and_target_columns": True' in source
    assert "fingerprint_geometry_loss" in source
    assert "latent_usage_gap" in source
    assert '"stop_before_flow"' in source


def test_stage_a_runner_is_bounded_and_uses_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "--epochs \"${SUCC_MOL_AE_EPOCHS:-4}\"" in run_source
    assert "--latent-tokens \"${SUCC_MOL_AE_LATENT_TOKENS:-16}\"" in run_source
    assert "01:00:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
