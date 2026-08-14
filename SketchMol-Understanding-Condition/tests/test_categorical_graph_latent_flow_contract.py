from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
FLOW_PATH = EXPERIMENT_DIR / "categorical_graph_latent_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_categorical_graph_latent_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_categorical_graph_latent_flow_pilot.sh"


def test_categorical_graph_flow_has_target_blind_direct_generation_contract() -> None:
    source = FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"generation_target_access": False' in source
    assert '"evaluation_target_access": True' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"oracle_reranking": False' in source
    assert '"valence_projection_or_repair": False' in source
    assert "class EquivariantGraphVelocity" in source
    assert "def sample_from_source" in source
    sample_source = source[source.index("def sample_from_source") : source.index("def finite_mean")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source


def test_categorical_graph_flow_runner_is_small_and_exact_n20() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_GRAPH_FLOW_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_GRAPH_FLOW_VALIDATION_LIMIT:-16}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
