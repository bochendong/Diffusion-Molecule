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
MODEL_PATH = EXPERIMENT_DIR / "source_relative_delta_diffusion.py"
RUN_PATH = EXPERIMENT_DIR / "run_source_relative_delta_diffusion_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_source_relative_delta_diffusion_pilot.sh"


def test_delta_diffusion_preserves_source_as_the_generative_base() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def delta_action_targets" in source
    assert "def apply_delta_actions" in source
    assert "def legal_node_action_mask" in source
    assert "def legal_edge_action_mask" in source
    assert '"source_relative_sparse_delta_diffusion": True' in source
    assert '"full_target_graph_diffusion": False' in source
    assert '"source_graph_exact_invariant_base": True' in source
    assert '"latent_usage_contrast": True' in source
    assert '"generation_target_access": False' in source
    assert '"posthoc_molecule_repair": False' in source
    assert '"valence_repair": False' in source
    sample = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample
    assert "target_example" not in sample
    assert "property_oracle" not in sample
    assert "MolFromSmiles" not in sample


def test_delta_diffusion_runner_is_matched_bounded_and_exact_n20() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_DELTA_DIFFUSION_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_DELTA_DIFFUSION_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_DELTA_DIFFUSION_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--diffusion-steps "${SUCC_DELTA_DIFFUSION_STEPS:-8}"' in run_source
    assert "--latent-usage-weight 0.20" in run_source
    assert "--latent-min-std 0.20" in run_source
    assert "--num-attempts 20" in run_source
    assert "--gate-validity 0.95" in run_source
    assert "--gate-mean-unique-valid 10" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=8G" in submit_source
    assert "00:20:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
