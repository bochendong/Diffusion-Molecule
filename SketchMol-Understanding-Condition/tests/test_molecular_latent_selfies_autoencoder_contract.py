from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
TRAIN_PATH = EXPERIMENT_DIR / "train_molecular_latent_selfies_autoencoder.py"
RUN_PATH = EXPERIMENT_DIR / "run_molecular_latent_selfies_autoencoder_v3.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_molecular_latent_selfies_autoencoder_v3.sh"


def test_selfies_gate_is_representation_only_and_warm_starts_latent() -> None:
    source = TRAIN_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"decoder_representation": "SELFIES"' in source
    assert '"condition_access": False' in source
    assert '"property_oracle_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"benchmark_generation_target_access": False' in source
    assert "--latent-checkpoint" in source
    assert "--resume-checkpoint" in source
    assert "load_state_dict(latent_checkpoint" in source
    assert "PositionalEncoding(latent_model.d_model, max_len=512)" in source
    assert '"stop_before_flow"' in source


def test_selfies_runner_keeps_fixed_gate_and_bounded_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "--epochs \"${SUCC_MOL_SELFIES_AE_EPOCHS:-12}\"" in run_source
    assert "molecular_latent_autoencoder_v2" in run_source
    assert "SUCC_MOL_SELFIES_AE_RESUME_CHECKPOINT" in run_source
    assert "01:00:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
