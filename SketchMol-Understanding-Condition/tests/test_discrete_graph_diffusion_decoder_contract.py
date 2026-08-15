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
MODEL_PATH = EXPERIMENT_DIR / "discrete_graph_diffusion_decoder.py"
RUN_PATH = EXPERIMENT_DIR / "run_discrete_graph_diffusion_decoder_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_discrete_graph_diffusion_decoder_pilot.sh"


def test_decoder_is_true_discrete_diffusion_and_target_blind_at_generation() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class JointGraphDenoiser" in source
    assert "class ContinuousDiscreteGraphDiffusion" in source
    assert "def corrupt_joint_states" in source
    assert "def remask_low_confidence" in source
    assert '"absorbing_discrete_graph_diffusion": True' in source
    assert '"joint_atom_state_diffusion": True' in source
    assert '"joint_bond_state_diffusion": True' in source
    assert '"train_only_state_vocabulary": True' in source
    assert '"generation_target_access": False' in source
    assert '"posthoc_molecule_repair": False' in source
    sample = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample
    assert "target_example" not in sample
    assert "property_oracle" not in sample
    assert "MolFromSmiles" not in sample


def test_runner_keeps_matched_bounded_exact_n20_contract() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_DISCRETE_DIFFUSION_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_DISCRETE_DIFFUSION_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_DISCRETE_DIFFUSION_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--diffusion-steps "${SUCC_DISCRETE_DIFFUSION_STEPS:-8}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "--gate-validity 0.95" in run_source
    assert "--gate-mean-unique-valid 10" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=8G" in submit_source
    assert "00:20:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
